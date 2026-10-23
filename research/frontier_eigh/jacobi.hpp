#pragma once

// Bounded research experiment, not a replacement for the standard solver.
// Included after Grid, Device, Phases and bulk_gemm in eigh.cpp.
namespace frontier_jacobi {
template<class Out, class In> __global__ void cast(Out* out, const In* in, size_t n) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<n) out[z]=Out(in[z]);
}
__global__ void scale_seed(float* a, size_t n, int exponent) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<n) a[z]=float(scalbn(double(a[z]),exponent));
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

// Local two-block workspace only. RefSyEv (Ogita--Aishima, 2018):
// R=I-U' U, H=U' S U, d_i=H_ii/(1-R_ii),
// E_ij=(H_ij+d_j R_ij)/(d_j-d_i), E_ii=R_ii/2, U <- U(I+E).
// Inside unresolved clusters use E_ij=R_ij/2, then a checked FP64
// Rayleigh--Ritz solve of that cluster, never division through a small gap.
struct PivotRefinement {
    Device<double> original, q, aq, gram, rayleigh, correction, next, cluster;
    Device<double> values, offdiag;
    Device<int> info;
    int seed_exponent=0;
    double original_norm=0;
    double seed_residual=0, residual=0, orthogonality=0, eigenvalue_change=0;
    std::string failure;
    explicit PivotRefinement(int m): original(size_t(m)*m), q(original.n),
        aq(original.n), gram(original.n), rayleigh(original.n),
        correction(original.n), next(original.n), cluster(original.n),
        values(m), offdiag(m), info(1) {}

    void prepare_seed(float* u, int m) {
        // Preserve the unscaled pivot for EVERY subsequent residual/Rayleigh
        // calculation. A globally ordinary-scale A can have near-zero local
        // pivots after Jacobi sweeps; normalize each FP32 seed independently.
        // ROCm 7.0.2 STEDC compares |rho*z| to 8*eps*max(max|D|,max|z|):
        // dimensionless z can otherwise impose an absolute deflation floor.
        cast<<<(size_t(m)*m+255)/256,256>>>(original.p,u,size_t(m)*m);
        HIP(hipGetLastError());
        auto s=original.get();
        double maximum=0, squared_norm=0;
        for (size_t z=0;z<size_t(m)*m;++z) {
            if (!std::isfinite(s[z])) throw std::runtime_error("nonfinite original Jacobi pivot");
            maximum=std::max(maximum,std::abs(s[z]));
            squared_norm+=s[z]*s[z];
        }
        original_norm=std::sqrt(squared_norm);
        seed_exponent=0;
        if (maximum!=0) std::frexp(maximum,&seed_exponent);
        if (seed_exponent) scale_seed<<<(size_t(m)*m+255)/256,256>>>(u,size_t(m)*m,-seed_exponent);
        HIP(hipGetLastError());
    }

    void restore_seed_values(float* d, int m) {
        if (seed_exponent) scale_seed<<<(m+255)/256,256>>>(d,m,seed_exponent);
        HIP(hipGetLastError());
    }

    void project(rocblas_handle h, int m, bool need_gram=true) {
        const double one=1, zero=0;
        BLAS(rocblas_dgemm(h,rocblas_operation_none,rocblas_operation_none,
            m,m,m,&one,original.p,m,q.p,m,&zero,aq.p,m));
        if (need_gram) BLAS(rocblas_dgemm(h,rocblas_operation_transpose,rocblas_operation_none,
            m,m,m,&one,q.p,m,q.p,m,&zero,gram.p,m));
        BLAS(rocblas_dgemm(h,rocblas_operation_transpose,rocblas_operation_none,
            m,m,m,&one,q.p,m,aq.p,m,&zero,rayleigh.p,m));
    }

    bool refine(rocblas_handle h, float* u, float* d, int m, int iterations,
                std::vector<unsigned long long>& counts, bool count) {
        failure.clear();
        seed_residual=residual=orthogonality=eigenvalue_change=0;
        if (count) ++counts[1]; // Unique attempted pair pivots, not both owners.
        cast<<<(size_t(m)*m+255)/256,256>>>(q.p,u,size_t(m)*m);
        HIP(hipGetLastError());
        std::vector<float> seed(m);
        HIP(hipMemcpy(seed.data(),d,m*sizeof(float),hipMemcpyDeviceToHost));
        const double snorm=original_norm;
        if (!std::isfinite(snorm)) { failure="nonfinite original pivot"; return false; }
        const double eps=std::numeric_limits<double>::epsilon();
        const double floor=256*eps*m;
        const double gate=std::max(floor,.25*double(std::numeric_limits<float>::epsilon()));
        std::vector<double> eigenvalues(m), initial(m);
        std::vector<int> cluster_id(m);
        std::vector<std::pair<int,int>> blocks;
        double seed_orth=0, gap_threshold=0;
        for (int step=0;step<iterations;++step) {
            project(h,m);
            auto g=gram.get(), r=rayleigh.get();
            double rnorm=0, offnorm=0;
            for (int j=0;j<m;++j) {
                if (!(g[j+size_t(j)*m]>0) || !std::isfinite(g[j+size_t(j)*m])) {
                    failure="invalid refinement Gram diagonal"; return false;
                }
                eigenvalues[j]=r[j+size_t(j)*m]/g[j+size_t(j)*m];
                if (!std::isfinite(eigenvalues[j])) { failure="nonfinite Rayleigh value"; return false; }
                for (int i=0;i<m;++i) {
                    size_t z=i+size_t(j)*m;
                    double defect=double(i==j)-g[z];
                    rnorm+=defect*defect;
                    if (i!=j) offnorm+=r[z]*r[z];
                }
            }
            if (!std::isfinite(rnorm) || !std::isfinite(offnorm)) {
                failure="nonfinite refinement projection"; return false;
            }
            if (!step) {
                auto x=q.get(), y=aq.get();
                double rr=0;
                for (int j=0;j<m;++j) for (int i=0;i<m;++i) {
                    size_t z=i+size_t(j)*m;
                    double v=y[z]-x[z]*double(seed[j]); rr+=v*v;
                }
                seed_residual=snorm==0 ? std::sqrt(rr) : std::sqrt(rr)/snorm;
                seed_orth=std::sqrt(rnorm);
                if (!std::isfinite(seed_residual) || seed_orth>=.1) {
                    failure="FP32 seed outside refinement neighborhood"; return false;
                }
                for (int j=0;j<m;++j) {
                    if (!std::isfinite(seed[j]) || (j && seed[j]<seed[j-1])) {
                        failure="FP32 seed eigenvalues nonfinite or unsorted"; return false;
                    }
                    initial[j]=double(seed[j]);
                }
                // Frobenius bounds are conservative: large clusters are reported,
                // not silently replaced by a full-pivot FP64 eigensolve.
                gap_threshold=16*(std::sqrt(offnorm)+snorm*seed_orth+eps*m*snorm);
                int first=0;
                for (int j=1;j<=m;++j) {
                    if (j<m && double(seed[j])-double(seed[j-1])<=gap_threshold) continue;
                    int id=int(blocks.size());
                    blocks.emplace_back(first,j);
                    for (int i=first;i<j;++i) cluster_id[i]=id;
                    first=j;
                }
            }
            std::vector<double> update(correction.n,0);
            for (int j=0;j<m;++j) for (int i=0;i<m;++i) {
                size_t z=i+size_t(j)*m;
                // Symmetrize products to avoid an arbitrary triangle bias.
                double rij=double(i==j)-.5*(g[z]+g[j+size_t(i)*m]);
                if (cluster_id[i]==cluster_id[j]) update[z]=.5*rij;
                else {
                    double gap=eigenvalues[j]-eigenvalues[i];
                    if (!std::isfinite(gap) || std::abs(gap)<=.5*gap_threshold) {
                        failure="new unresolved cross-cluster gap; no fallback"; return false;
                    }
                    update[z]=(.5*(r[z]+r[j+size_t(i)*m])+eigenvalues[j]*rij)/gap;
                }
                if (i==j) update[z]+=1;
                if (!std::isfinite(update[z])) { failure="nonfinite Newton correction"; return false; }
            }
            correction.put(update);
            const double one=1, zero=0;
            BLAS(rocblas_dgemm(h,rocblas_operation_none,rocblas_operation_none,
                m,m,m,&one,q.p,m,correction.p,m,&zero,next.p,m));
            HIP(hipMemcpyAsync(q.p,next.p,size_t(m)*m*sizeof(double),hipMemcpyDeviceToDevice));
            // Recompute H after the orthogonality/Newton update; using the old
            // H here would solve the wrong Rayleigh--Ritz problem.
            std::vector<double> projected;
            if (blocks.size()<size_t(m)) {
                project(h,m,false);
                projected=rayleigh.get();
            }
            for (auto block:blocks) {
                int first=block.first, k=block.second-first;
                if (k==1) continue;
                std::vector<double> small(size_t(k)*k);
                for (int j=0;j<k;++j) for (int i=0;i<k;++i)
                    small[i+size_t(j)*k]=.5*(projected[first+i+size_t(first+j)*m]
                        +projected[first+j+size_t(first+i)*m]);
                HIP(hipMemcpy(cluster.p,small.data(),small.size()*sizeof(double),hipMemcpyHostToDevice));
                BLAS(rocsolver_dsyevd(h,rocblas_evect_original,rocblas_fill_lower,
                    k,cluster.p,k,values.p,offdiag.p,info.p));
                if (info.get()[0]!=0) { failure="FP64 cluster Rayleigh--Ritz info != 0"; return false; }
                if (count) {
                    ++counts[k]; // Counts actual solves, once per pair.
                    if (k==m) ++counts[0]; // Explicitly disclose full-pivot clusters.
                }
                BLAS(rocblas_dgemm(h,rocblas_operation_none,rocblas_operation_none,
                    m,k,k,&one,q.p+size_t(first)*m,m,cluster.p,k,&zero,next.p,m));
                HIP(hipMemcpyAsync(q.p+size_t(first)*m,next.p,size_t(m)*k*sizeof(double),hipMemcpyDeviceToDevice));
            }
        }
        project(h,m);
        auto g=gram.get(), r=rayleigh.get(), x=q.get(), y=aq.get();
        double rr=0, gg=0;
        for (int j=0;j<m;++j) {
            if (!(g[j+size_t(j)*m]>0)) { failure="invalid final Gram diagonal"; return false; }
            eigenvalues[j]=r[j+size_t(j)*m]/g[j+size_t(j)*m];
            if (!std::isfinite(eigenvalues[j])) { failure="nonfinite final Rayleigh value"; return false; }
            for (int i=0;i<m;++i) {
                size_t z=i+size_t(j)*m;
                double v=y[z]-x[z]*eigenvalues[j]; rr+=v*v;
                double w=g[z]-double(i==j); gg+=w*w;
            }
        }
        residual=snorm==0 ? std::sqrt(rr) : std::sqrt(rr)/snorm;
        orthogonality=std::sqrt(gg);
        auto sorted=eigenvalues;
        std::sort(sorted.begin(),sorted.end());
        std::sort(initial.begin(),initial.end());
        eigenvalue_change=0;
        for (int j=0;j<m;++j) eigenvalue_change=std::max(eigenvalue_change,std::abs(sorted[j]-initial[j]));
        double eigenvalue_gate=(8*(seed_residual+seed_orth)+floor)*snorm;
        if (!std::isfinite(residual) || !std::isfinite(orthogonality)
            || !std::isfinite(eigenvalue_change) || residual>gate || orthogonality>gate
            || eigenvalue_change>eigenvalue_gate) {
            failure="refinement original-pivot residual/orthogonality/eigenvalue gate";
            return false;
        }
        // Values correspond to the returned columns, not their sorted copy.
        HIP(hipMemcpy(values.p,eigenvalues.data(),m*sizeof(double),hipMemcpyHostToDevice));
        cast<<<(size_t(m)*m+255)/256,256>>>(u,q.p,size_t(m)*m);
        cast<<<(m+255)/256,256>>>(d,values.p,m);
        HIP(hipGetLastError());
        return true;
    }
};
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
    if (jacobi_refinement<0 || jacobi_refinement>2)
        throw std::runtime_error("Jacobi refinement must be 0, 1 or 2");
    if (jacobi_refinement && (std::is_same_v<T,double> || jacobi_pivot64 || jacobi_local!="dc"))
        throw std::runtime_error("Jacobi refinement requires FP32 state, pivot64=0 and local=dc");
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
    Device<T> d(2*b), e(2*b); Device<int> info(1), local_sweeps(1);
    Device<double> precise(jacobi_pivot64 ? u.n : 1), precise_d(2*b), precise_e(2*b);
    std::unique_ptr<PivotRefinement> refinement;
    if (jacobi_refinement) refinement=std::make_unique<PivotRefinement>(2*b);
    std::vector<unsigned long long> cluster_counts(2*b+1,0);
    double refinement_time=0, seed_time=0;
    double max_seed_residual=0, max_pivot_residual=0, max_pivot_orth=0, max_eval_change=0, max_seed_exponent=0;
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
            if constexpr(std::is_same_v<T,double>) {
                if (jacobi_local=="jacobi") BLAS(rocsolver_dsyevj(h,rocblas_esort_ascending,
                    rocblas_evect_original,rocblas_fill_lower,m,u.p,m,0,e.p,30,local_sweeps.p,d.p,info.p));
                else BLAS(rocsolver_dsyevd(h,rocblas_evect_original,
                    rocblas_fill_lower,m,u.p,m,d.p,e.p,info.p));
            }
            else if (jacobi_pivot64) {
                cast<<<(size_t(m)*m+255)/256,256>>>(precise.p,u.p,size_t(m)*m);
                if (jacobi_local=="jacobi") BLAS(rocsolver_dsyevj(h,rocblas_esort_ascending,
                    rocblas_evect_original,rocblas_fill_lower,m,precise.p,m,0,precise_e.p,30,local_sweeps.p,precise_d.p,info.p));
                else BLAS(rocsolver_dsyevd(h,rocblas_evect_original,
                    rocblas_fill_lower,m,precise.p,m,precise_d.p,precise_e.p,info.p));
                cast<<<(size_t(m)*m+255)/256,256>>>(u.p,precise.p,size_t(m)*m);
                cast<<<(m+255)/256,256>>>(d.p,precise_d.p,m);
            } else {
                if (refinement) refinement->prepare_seed(u.p,m);
                if (jacobi_local=="jacobi") BLAS(rocsolver_ssyevj(h,rocblas_esort_ascending,
                    rocblas_evect_original,rocblas_fill_lower,m,u.p,m,0,e.p,30,local_sweeps.p,d.p,info.p));
                else BLAS(rocsolver_ssyevd(h,rocblas_evect_original,
                    rocblas_fill_lower,m,u.p,m,d.p,e.p,info.p));
                if (refinement) refinement->restore_seed_values(d.p,m);
            }
            int bad=info.get()[0]!=0;
            MPI(MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,g.row));
            if (bad) throw std::runtime_error("Jacobi local eigensolver info != 0");
            if constexpr(std::is_same_v<T,float>) {
                if (refinement) {
                    seed_time+=MPI_Wtime()-t;
                    max_seed_exponent=std::max(max_seed_exponent,std::abs(double(refinement->seed_exponent)));
                    double start=MPI_Wtime();
                    bad=!refinement->refine(h,u.p,d.p,m,jacobi_refinement,cluster_counts,g.rank==lo);
                    sync();
                    refinement_time+=MPI_Wtime()-start;
                    if (bad) std::fprintf(stderr,
                        "JACOBI_REFINE_FAIL rank=%d sweep=%d round=%d m=%d reason=%s seed_residual=%.17g residual=%.17g orth=%.17g eigenvalue_change=%.17g pivot_norm=%.17g seed_scale_exponent=%d\n",
                        g.rank,sweep,round,m,refinement->failure.c_str(),refinement->seed_residual,
                        refinement->residual,refinement->orthogonality,refinement->eigenvalue_change,
                        refinement->original_norm,refinement->seed_exponent);
                    MPI(MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,g.row));
                    if (bad) throw std::runtime_error("Jacobi pivot refinement rejected; no fallback");
                    max_seed_residual=std::max(max_seed_residual,refinement->seed_residual);
                    max_pivot_residual=std::max(max_pivot_residual,refinement->residual);
                    max_pivot_orth=std::max(max_pivot_orth,refinement->orthogonality);
                    max_eval_change=std::max(max_eval_change,refinement->eigenvalue_change);
                }
            }
            normalize_columns<<<m,256>>>(u.p,m);
            if (refinement) sync(); // Account for the final cast/normalization in pivot time.
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
        if (refinement) {
            double seed=g.maximum(seed_time), refined=g.maximum(refinement_time);
            double sr=g.maximum(max_seed_residual), rr=g.maximum(max_pivot_residual);
            double oo=g.maximum(max_pivot_orth), ee=g.maximum(max_eval_change);
            double exponent=g.maximum(max_seed_exponent);
            std::vector<unsigned long long> counts(cluster_counts.size());
            MPI(MPI_Allreduce(cluster_counts.data(),counts.data(),int(counts.size()),
                MPI_UNSIGNED_LONG_LONG,MPI_SUM,g.row));
            if (!g.rank) {
                std::printf("JACOBI_REFINE sweep=%d iterations=%d seed_seconds_cumulative=%.9g refinement_seconds_cumulative=%.9g seed_residual_max=%.17g pivot_residual_max=%.17g pivot_orth_raw_max=%.17g eigenvalue_change_abs_max=%.17g seed_scale_exponent_abs_max=%.0f pivots_cumulative=%llu full_pivot_cluster_solves_cumulative=%llu clusters_cumulative=size:solves",
                    sweep,jacobi_refinement,seed,refined,sr,rr,oo,ee,exponent,counts[1],counts[0]);
                for (int k=2;k<int(counts.size());++k) if (counts[k]) std::printf(" %d:%llu",k,counts[k]);
                std::printf("\n");
            }
        }
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
    if (!g.rank) std::printf("JACOBI_DONE pivot64=%d local=%s refine=%d offdiag_rel=%.17g\n",int(jacobi_pivot64),jacobi_local.c_str(),jacobi_refinement,ratio);
    return times;
}
