#pragma once

// Bounded research experiment, not a replacement for the standard solver.
// Included after Grid, Device, Phases and bulk_gemm in eigh.cpp.
namespace frontier_jacobi {
template<class Out, class In> __global__ void cast(Out* out, const In* in, size_t n) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<n) out[z]=Out(in[z]);
}
template<class T> __global__ void identity(T* q, int n, int w, int start) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(n)*w) q[z]=T(int(z%n)==start+int(z/n));
}
template<class T> __global__ void pivot(T* u, const T* panels, int n,
                                      int lo, int hi, int wl, int m) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(m)*m) {
        int i=z%m, j=z/m;
        int ri=i<wl?lo+i:hi+i-wl, rj=j<wl?lo+j:hi+j-wl;
        u[z]=T(.5*double(panels[ri+size_t(j)*n])+
               .5*double(panels[rj+size_t(i)*n]));
    }
}
template<class T> __global__ void diagonal_pivot(T* a, const T* d, int n, int w,
    int lo, int hi, int wl, int m, int offset) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(m)*w) {
        int i=z%m, j=z/m, row=i<wl?lo+i:hi+i-wl;
        a[row+size_t(j)*n]=i==offset+j ? d[i] : T(0);
    }
}
template<class T> __global__ void rows(T* packed, T* a, int n, int w,
                                     int lo, int hi, int wl, int m, bool unpack) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(m)*w) {
        int i=z%m, r=i<wl?lo+i:hi+i-wl;
        size_t k=r+(z/m)*n;
        if (unpack) a[k]=packed[z]; else packed[z]=a[k];
    }
}
template<class T> __global__ void normalize_columns(T* q, int n) {
    __shared__ double partial[256];
    int t=threadIdx.x, j=blockIdx.x;
    double sum=0;
    for (int i=t;i<n;i+=256) { double x=q[i+size_t(j)*n]; sum+=x*x; }
    partial[t]=sum; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (t<stride) partial[t]+=partial[t+stride];
        __syncthreads();
    }
    double inverse=1/std::sqrt(partial[0]);
    for (int i=t;i<n;i+=256) q[i+size_t(j)*n]=T(double(q[i+size_t(j)*n])*inverse);
}
template<class T> void retile(Grid& src, Grid& dst, Device<T>& a, Device<T>& b) {
    auto local=a.get();
    std::vector<double> in(local.begin(),local.end()), out(size_t(dst.nr)*dst.nc);
    int one=1;
    pdgemr2d_(&src.n,&src.n,in.data(),&one,&one,src.desc,
              out.data(),&one,&one,dst.desc,&src.ctxt);
    b.put(std::vector<T>(out.begin(),out.end()));
}
}

template<class T> Phases jacobi(Grid& original, rocblas_handle h, Device<T>& input,
    Device<T>& output, std::vector<double>& eval, int max_sweeps, double tol) {
    using namespace frontier_jacobi;
    const int n=original.n, p=original.pr*original.pc, b=(n+p-1)/p;
    if (p<4 || p%2) throw std::runtime_error("Jacobi requires even P >= 4; P=2 would gather full A");
    if (max_sweeps<1 || !std::isfinite(tol) || tol<=0)
        throw std::runtime_error("Jacobi requires positive finite tolerance and max_sweeps >= 1");
    // Grid currently forbids empty local dimensions, including in this layout.
    if ((p-1)*b>=n) throw std::runtime_error("Jacobi ceil(n/P) layout requires a nonempty final rank block");
    if constexpr(std::is_same_v<T,double>) {
        if (half_bulk || bf16_bulk) throw std::runtime_error("Jacobi mixed GEMM requires FP32 state");
    }
    HIP(hipDeviceSynchronize());
    BLAS(rocblas_set_stream(h,nullptr));
    Phases times;
    std::unique_ptr<Grid> internal;
    bool change=original.pr!=1 || original.pc!=p || original.b!=b || !original.rccl || !original.gpu_mpi;
    double begin=original.tick();
    if (change) internal=std::make_unique<Grid>(n,b,1,p,true,true);
    Grid& g=internal?*internal:original;
    std::unique_ptr<Device<T>> aa, qq;
    if (change) {
        aa=std::make_unique<Device<T>>(size_t(n)*g.nc);
        qq=std::make_unique<Device<T>>(size_t(n)*g.nc);
        retile(original,g,input,*aa);
    }
    Device<T>& a=change?*aa:input;
    Device<T>& q=change?*qq:output;
    identity<<<(q.n+255)/256,256>>>(q.p,n,g.nc,g.rank*b);
    HIP(hipGetLastError());
    times.retile=original.tick()-begin;
    auto width=[&](int r) { return std::min(b,n-r*b); };
    // At most two local column panels and a two-block pivot, never a dense gather.
    Device<T> pa(size_t(n)*2*b), pq(pa.n), u(size_t(2*b)*2*b), broadcast(u.n);
    Device<T> scratch(size_t(n)*g.nc), packed(size_t(2*b)*g.nc), left(packed.n);
    Device<T> d(2*b), e(2*b); Device<int> info(1);
    Device<double> precise(jacobi_pivot64 ? u.n : 1), precise_d(2*b), precise_e(2*b);
    double solve_time=0, rotation_time=0, communication=0;
    auto sync=[] { HIP(hipDeviceSynchronize()); };
    auto metric=[&]() {
        // Host-local FP64 accumulation; only two scalars are globally reduced.
        auto host=a.get(); double sums[2]={};
        for (int j=0;j<g.nc;++j) for (int i=0;i<n;++i) {
            double x=double(host[i+size_t(j)*n]);
            sums[1]+=x*x;
            if (i!=g.rank*b+j) sums[0]+=x*x;
        }
        double t=MPI_Wtime();
        MPI(MPI_Allreduce(MPI_IN_PLACE,sums,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));
        communication+=MPI_Wtime()-t;
        if (!std::isfinite(sums[0]) || !std::isfinite(sums[1]))
            throw std::runtime_error("Jacobi nonfinite Frobenius metric");
        return sums[1]==0 ? 0. : std::sqrt(sums[0]/sums[1]);
    };
    double ratio=metric();
    if (!g.rank) std::printf("JACOBI sweep=0 offdiag_rel=%.17g\n",ratio);
    begin=g.tick();
    for (int sweep=1;ratio>tol && sweep<=max_sweeps;++sweep) {
        double sweep_start=MPI_Wtime();
        std::vector<int> circle(p);
        for (int r=0;r<p;++r) circle[r]=r;
        for (int round=0;round<p-1;++round) {
            int partner=-1;
            for (int k=0;k<p/2;++k) {
                int x=circle[k], y=circle[p-1-k];
                if (x==g.rank) partner=y;
                if (y==g.rank) partner=x;
            }
            int lo=std::min(g.rank,partner), hi=std::max(g.rank,partner);
            int wl=width(lo), m=wl+width(hi), offset=g.rank==lo?0:wl;
            double t=MPI_Wtime();
            HIP(hipMemcpyAsync(pa.p+size_t(offset)*n,a.p,a.n*sizeof(T),hipMemcpyDeviceToDevice));
            HIP(hipMemcpyAsync(pq.p+size_t(offset)*n,q.p,q.n*sizeof(T),hipMemcpyDeviceToDevice));
            sync(); // GPU-aware MPI must not observe pending default-stream writes.
            int other_offset=g.rank==lo?wl:0;
            MPI(MPI_Sendrecv(a.p,int(a.n),Grid::type<T>(),partner,170,
                pa.p+size_t(other_offset)*n,n*width(partner),Grid::type<T>(),partner,170,g.row,MPI_STATUS_IGNORE));
            MPI(MPI_Sendrecv(q.p,int(q.n),Grid::type<T>(),partner,171,
                pq.p+size_t(other_offset)*n,n*width(partner),Grid::type<T>(),partner,171,g.row,MPI_STATUS_IGNORE));
            communication+=MPI_Wtime()-t;
            t=MPI_Wtime();
            pivot<<<(size_t(m)*m+255)/256,256>>>(u.p,pa.p,n,lo*b,hi*b,wl,m);
            HIP(hipGetLastError());
            if constexpr(std::is_same_v<T,double>) BLAS(rocsolver_dsyevd(h,rocblas_evect_original,
                rocblas_fill_lower,m,u.p,m,d.p,e.p,info.p));
            else if (jacobi_pivot64) {
                cast<<<(size_t(m)*m+255)/256,256>>>(precise.p,u.p,size_t(m)*m);
                BLAS(rocsolver_dsyevd(h,rocblas_evect_original,
                    rocblas_fill_lower,m,precise.p,m,precise_d.p,precise_e.p,info.p));
                cast<<<(size_t(m)*m+255)/256,256>>>(u.p,precise.p,size_t(m)*m);
                cast<<<(m+255)/256,256>>>(d.p,precise_d.p,m);
            } else BLAS(rocsolver_ssyevd(h,rocblas_evect_original,
                rocblas_fill_lower,m,u.p,m,d.p,e.p,info.p));
            int bad=info.get()[0]!=0;
            MPI(MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,g.row));
            if (bad) throw std::runtime_error("Jacobi local SYEVD info != 0");
            normalize_columns<<<m,256>>>(u.p,m);
            solve_time+=MPI_Wtime()-t;
            // Freeze all rotations before the two-sided update.
            t=MPI_Wtime();
            bulk_gemm(h,false,n,g.nc,m,T(1),pa.p,n,u.p+size_t(offset)*m,m,T(0),scratch.p,n);
            HIP(hipMemcpyAsync(a.p,scratch.p,a.n*sizeof(T),hipMemcpyDeviceToDevice));
            bulk_gemm(h,false,n,g.nc,m,T(1),pq.p,n,u.p+size_t(offset)*m,m,T(0),scratch.p,n);
            HIP(hipMemcpyAsync(q.p,scratch.p,q.n*sizeof(T),hipMemcpyDeviceToDevice));
            sync();
            rotation_time+=MPI_Wtime()-t;
            t=MPI_Wtime();
            MPI(MPI_Barrier(g.row)); // finish EVERY right update before any left update
            communication+=MPI_Wtime()-t;
            for (int k=0;k<p/2;++k) {
                int x=std::min(circle[k],circle[p-1-k]), y=std::max(circle[k],circle[p-1-k]);
                int wx=width(x), mm=wx+width(y);
                t=MPI_Wtime();
                if (g.rank==x) HIP(hipMemcpyAsync(broadcast.p,u.p,size_t(mm)*mm*sizeof(T),hipMemcpyDeviceToDevice));
                CCL(ncclBroadcast(broadcast.p,broadcast.p,size_t(mm)*mm,
                    std::is_same_v<T,double>?ncclFloat64:ncclFloat32,x,g.ccl_row,nullptr));
                sync(); communication+=MPI_Wtime()-t;
                t=MPI_Wtime();
                rows<<<(size_t(mm)*g.nc+255)/256,256>>>(packed.p,a.p,n,g.nc,x*b,y*b,wx,mm,false);
                bulk_gemm(h,true,mm,g.nc,mm,T(1),broadcast.p,mm,packed.p,mm,T(0),left.p,mm);
                rows<<<(size_t(mm)*g.nc+255)/256,256>>>(left.p,a.p,n,g.nc,x*b,y*b,wx,mm,true);
                HIP(hipGetLastError()); sync(); rotation_time+=MPI_Wtime()-t;
            }
            // In exact arithmetic this principal block is diag(d). Avoid retaining
            // GEMM cancellation noise as the next sweep's off-diagonal floor.
            diagonal_pivot<<<(size_t(m)*g.nc+255)/256,256>>>(a.p,d.p,n,g.nc,lo*b,hi*b,wl,m,offset);
            HIP(hipGetLastError());
            int last=circle.back();
            for (int k=p-1;k>1;--k) circle[k]=circle[k-1];
            circle[1]=last;
        }
        ratio=metric();
        double elapsed=g.maximum(MPI_Wtime()-sweep_start);
        double st=g.maximum(solve_time), rt=g.maximum(rotation_time), ct=g.maximum(communication);
        if (!g.rank) std::printf("JACOBI sweep=%d offdiag_rel=%.17g sweep_seconds=%.9g local_solve_cumulative=%.9g rotation_cumulative=%.9g communication_cumulative=%.9g\n",
            sweep,ratio,elapsed,st,rt,ct);
    }
    times.reduction=g.tick()-begin;
    if (ratio>tol) throw std::runtime_error("Jacobi max_sweeps exhausted without convergence (see offdiag history; no fallback)");
    begin=g.tick();
    auto host=a.get(); eval.assign(n,0);
    for (int j=0;j<g.nc;++j) eval[g.rank*b+j]=double(host[g.rank*b+j+size_t(j)*n]);
    MPI(MPI_Allreduce(MPI_IN_PLACE,eval.data(),n,MPI_DOUBLE,MPI_SUM,g.row));
    for (double x:eval) if (!std::isfinite(x)) throw std::runtime_error("Jacobi nonfinite eigenvalue");
    std::vector<int> order(n), destination(n);
    for (int j=0;j<n;++j) order[j]=j;
    std::stable_sort(order.begin(),order.end(),[&](int x,int y) { return eval[x]<eval[y]; });
    std::vector<double> sorted(n);
    for (int j=0;j<n;++j) { destination[order[j]]=j; sorted[j]=eval[order[j]]; }
    eval.swap(sorted);
    // Pack columns in increasing old-global-column order for each destination.
    // Receivers reconstruct that same order, so no per-column metadata is needed.
    auto hq=q.get();
    std::vector<int> sc(p,0), rc(p,0), sd(p,0), rd(p,0);
    for (int j=0;j<g.nc;++j) sc[destination[g.rank*b+j]/b]+=n;
    MPI(MPI_Alltoall(sc.data(),1,MPI_INT,rc.data(),1,MPI_INT,g.row));
    for (int r=1;r<p;++r) { sd[r]=sd[r-1]+sc[r-1]; rd[r]=rd[r-1]+rc[r-1]; }
    std::vector<T> send(hq.size()), receive(hq.size());
    auto cursor=sd;
    for (int j=0;j<g.nc;++j) {
        int r=destination[g.rank*b+j]/b;
        std::copy_n(hq.data()+size_t(j)*n,n,send.data()+cursor[r]); cursor[r]+=n;
    }
    MPI(MPI_Alltoallv(send.data(),sc.data(),sd.data(),Grid::type<T>(),
        receive.data(),rc.data(),rd.data(),Grid::type<T>(),g.row));
    cursor=rd;
    for (int old=0;old<n;++old) if (destination[old]/b==g.rank) {
        int source=old/b, j=destination[old]-g.rank*b;
        std::copy_n(receive.data()+cursor[source],n,hq.data()+size_t(j)*n); cursor[source]+=n;
    }
    q.put(hq);
    times.back=g.tick()-begin;
    if (change) {
        begin=g.tick();
        retile(g,original,q,output);
        // Preserve the transformed A in the caller's original distribution too.
        retile(g,original,a,input);
        times.retile+=g.tick()-begin;
        original.comm+=g.comm;
    }
    original.comm+=communication;
    if (!g.rank) std::printf("JACOBI_DONE pivot64=%d offdiag_rel=%.17g\n",int(jacobi_pivot64),ratio);
    return times;
}
