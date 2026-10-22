// Standalone research solver: real symmetric, full eigenvectors, 2D block cyclic.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <rccl/rccl.h>
#include <elpa/elpa.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {
void Cblacs_get(int, int, int*);
void Cblacs_gridinit(int*, const char*, int, int);
void Cblacs_gridexit(int);
void descinit_(int*, const int*, const int*, const int*, const int*, const int*,
               const int*, const int*, const int*, int*);
void pdstedc_(const char*, const int*, double*, double*, double*, const int*,
              const int*, const int*, double*, const int*, int*, const int*, int*);
void pdtran_(const int*, const int*, const double*, const double*, const int*,
             const int*, const int*, const double*, double*, const int*, const int*, const int*);
void dsyev_(const char*, const char*, const int*, double*, const int*, double*,
            double*, const int*, int*);
}

#define HIP(x) do { auto s = (x); if (s != hipSuccess) throw std::runtime_error( \
    std::string(#x) + ": " + hipGetErrorString(s)); } while (0)
#define BLAS(x) do { auto s = (x); if (s != rocblas_status_success) throw std::runtime_error( \
    std::string(#x) + ": " + std::to_string(s)); } while (0)
#define MPI(x) do { int s = (x); if (s != MPI_SUCCESS) throw std::runtime_error(#x); } while (0)
#define CCL(x) do { auto s = (x); if (s != ncclSuccess) throw std::runtime_error( \
    std::string(#x) + ": " + ncclGetErrorString(s)); } while (0)

template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count): n(count) { HIP(hipMalloc(&p, std::max(size_t(1), n)*sizeof(T))); }
    ~Device() { hipFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& h) { HIP(hipMemcpy(p, h.data(), n*sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> h(n); HIP(hipMemcpy(h.data(), p, n*sizeof(T), hipMemcpyDeviceToHost)); return h;
    }
};

struct Grid {
    int n, b, pr, pc, rank, r, c, nr, nc, ctxt, desc[9];
    MPI_Comm row, col;
    std::vector<int> rows, cols;
    double comm = 0;
    bool gpu_mpi;
    bool rccl;
    ncclComm_t ccl_row=nullptr, ccl_col=nullptr;
    Grid(int n_, int b_, int pr_, int pc_, bool gm, bool ccl): n(n_), b(b_), pr(pr_), pc(pc_), gpu_mpi(gm), rccl(ccl) {
        MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        int size; MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
        if (pr*pc != size || n < 2 || b < 1 || pr < 1 || pc < 1)
            throw std::runtime_error("invalid size, block, or grid");
        r = rank/pc; c = rank%pc;
        MPI(MPI_Comm_split(MPI_COMM_WORLD, r, c, &row));
        MPI(MPI_Comm_split(MPI_COMM_WORLD, c, r, &col));
        if (rccl) {
            if (!gpu_mpi) throw std::runtime_error("RCCL experiment requires --gpu-mpi 1");
            auto init=[&](MPI_Comm comm_,int myrank,int size_,ncclComm_t* result) {
                ncclUniqueId id;
                if (!myrank) CCL(ncclGetUniqueId(&id));
                MPI(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,comm_));
                CCL(ncclCommInitRank(result,size_,id,myrank));
            };
            init(row,c,pc,&ccl_row); init(col,r,pr,&ccl_col);
        }
        for (int i=0; i<n; ++i) {
            if ((i/b)%pr == r) rows.push_back(i);
            if ((i/b)%pc == c) cols.push_back(i);
        }
        nr = rows.size(); nc = cols.size();
        // Keep the initial experiment's local dimensions nonzero; arbitrary rectangular grids remain supported.
        if (!nr || !nc) throw std::runtime_error("choose n/block so every rank owns rows and columns");
        Cblacs_get(-1, 0, &ctxt); Cblacs_gridinit(&ctxt, "R", pr, pc);
        int zero=0, info;
        descinit_(desc, &n, &n, &b, &b, &zero, &zero, &ctxt, &nr, &info);
        if (info) throw std::runtime_error("descinit failed");
    }
    ~Grid() {
        if (rccl) { CCL(ncclCommDestroy(ccl_row)); CCL(ncclCommDestroy(ccl_col)); }
        Cblacs_gridexit(ctxt); MPI_Comm_free(&row); MPI_Comm_free(&col);
    }
    int li(int i, int p) const { return (i/b/p)*b + i%b; }
    template<class T> static MPI_Datatype type() { return std::is_same_v<T,double> ? MPI_DOUBLE : MPI_FLOAT; }
    template<class T> void sum(T* p, int count, MPI_Comm comm_) {
        double t=MPI_Wtime(); MPI(MPI_Allreduce(MPI_IN_PLACE,p,count,type<T>(),MPI_SUM,comm_)); comm+=MPI_Wtime()-t;
    }
    template<class T> void bcast(T* p, int count, int root, MPI_Comm comm_) {
        double t=MPI_Wtime(); MPI(MPI_Bcast(p,count,type<T>(),root,comm_)); comm+=MPI_Wtime()-t;
    }
    template<class T> void device_sum(Device<T>& d, MPI_Comm comm_) {
        HIP(hipDeviceSynchronize());
        if (rccl) {
            double t=MPI_Wtime();
            CCL(ncclAllReduce(d.p,d.p,d.n,std::is_same_v<T,double>?ncclFloat64:ncclFloat32,
                ncclSum,comm_==row?ccl_row:ccl_col,nullptr));
            HIP(hipStreamSynchronize(nullptr)); comm+=MPI_Wtime()-t;
        } else if (gpu_mpi) sum(d.p, d.n, comm_);
        else { auto h=d.get(); sum(h.data(), h.size(), comm_); d.put(h); }
    }
    template<class T> void device_bcast(T* p, int count, int root, MPI_Comm comm_) {
        if (!rccl) { bcast(p,count,root,comm_); return; }
        double t=MPI_Wtime();
        CCL(ncclBroadcast(p,p,count,std::is_same_v<T,double>?ncclFloat64:ncclFloat32,
            root,comm_==row?ccl_row:ccl_col,nullptr));
        HIP(hipStreamSynchronize(nullptr)); comm+=MPI_Wtime()-t;
    }
    double tick() { HIP(hipDeviceSynchronize()); MPI(MPI_Barrier(MPI_COMM_WORLD)); return MPI_Wtime(); }
    double maximum(double x) { double y; MPI(MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD)); return y; }
};

template<class T> void gemv(rocblas_handle h, bool trans, int m, int n, int lda,
                            T* a, T* x, T* y, T alpha=T(1), T beta=T(0)) {
    auto op=trans?rocblas_operation_transpose:rocblas_operation_none;
    if constexpr(std::is_same_v<T,double>) BLAS(rocblas_dgemv(h,op,m,n,&alpha,a,lda,x,1,&beta,y,1));
    else BLAS(rocblas_sgemv(h,op,m,n,&alpha,a,lda,x,1,&beta,y,1));
}

template<class T> void gemm(rocblas_handle h, bool trans, int m, int n, int k,
    T alpha, T* a, int lda, T* b, int ldb, T beta, T* c, int ldc, bool transb=false) {
    auto op=trans?rocblas_operation_transpose:rocblas_operation_none;
    auto opb=transb?rocblas_operation_transpose:rocblas_operation_none;
    if constexpr(std::is_same_v<T,double>) BLAS(rocblas_dgemm(h,op,opb,
        m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc));
    else BLAS(rocblas_sgemm(h,op,opb,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc));
}

int half_bulk=0;
template<class T> __global__ void to_half(__half* out, const T* in, int rows, int cols, int ld, int parts) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(rows)*cols) {
        float x=float(in[z%rows+(z/rows)*ld]);
        // MI200 MFMA flushes FP16 subnormals. Route them through the scaled low part.
        out[z]=__float2half(parts==2 && fabsf(x)<0x1p-14f ? 0.f : x);
        if (parts==2) out[z+size_t(rows)*cols]=__float2half((x-__half2float(out[z]))*4096.f);
    }
}
template<class T> void bulk_gemm(rocblas_handle h, bool trans, int m, int n, int k,
    T alpha, T* a, int lda, T* b, int ldb, T beta, T* c, int ldc, bool transb=false) {
    if (!half_bulk) { gemm(h,trans,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,transb); return; }
    int ar=trans?k:m, ac=trans?m:k, br=transb?n:k, bc=transb?k:n;
    size_t an=size_t(ar)*ac, bn=size_t(br)*bc;
    Device<__half> ah(an*half_bulk), bh(bn*half_bulk);
    to_half<<<(an+255)/256,256>>>(ah.p,a,ar,ac,lda,half_bulk);
    to_half<<<(bn+255)/256,256>>>(bh.p,b,br,bc,ldb,half_bulk);
    HIP(hipGetLastError());
    // Two-part mode omits only A_low B_low; accumulators and matrix state remain FP32.
    for (int term=0;term<(half_bulk==2?3:1);++term) {
        T add=term?T(1):beta, factor=term?alpha/T(4096):alpha;
        BLAS(rocblas_gemm_ex(h,trans?rocblas_operation_transpose:rocblas_operation_none,
            transb?rocblas_operation_transpose:rocblas_operation_none,m,n,k,
            &factor,ah.p+(term==2?an:0),rocblas_datatype_f16_r,ar,
            bh.p+(term==1?bn:0),rocblas_datatype_f16_r,br,
            &add,c,rocblas_datatype_f32_r,ldc,c,rocblas_datatype_f32_r,ldc,
            rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
    }
}

template<class T> __global__ void rank2(T* a, const T* v, const T* w,
    const int* rows, const int* cols, int nr, int mr, int mc, int r0, int c0) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z>=size_t(mr)*mc) return;
    int ri=r0+z%mr, cj=c0+z/mr;
    int i=rows[ri], j=cols[cj];
    a[ri+size_t(cj)*nr] -= v[i]*w[j] + w[i]*v[j];
}

template<class T> __global__ void rank1(T* q, const T* v, const T* y,
    const int* rows, int nr, int nc, T tau) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(nr)*nc) q[z] -= tau*v[rows[z%nr]]*y[z/nr];
}

template<class T> __global__ void pack_panel(T* u, T* d, const T* v, const T* w,
    const int* rows, const int* cols, int nr, int nc, int pos, int block) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<nr) { u[i+size_t(pos)*nr]=v[rows[i]]; u[i+size_t(pos+block)*nr]=w[rows[i]]; }
    if (i<nc) { d[i+size_t(pos)*nc]=w[cols[i]]; d[i+size_t(pos+block)*nc]=v[cols[i]]; }
}

template<class T> __global__ void correct_column(T* a, const T* u, const T* d,
    int nr, int nc, int localcol, int block) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=nr) return;
    T x=0;
    for (int p=0;p<2*block;++p) x+=u[i+size_t(p)*nr]*d[localcol+size_t(p)*nc];
    a[i+size_t(localcol)*nr]-=x;
}

template<class T> std::vector<T> column(Grid& g, Device<T>& a, int k) {
    std::vector<T> local(g.nr), full(g.n,0);
    int owner=(k/g.b)%g.pc;
    if (g.c==owner) HIP(hipMemcpy(local.data(), a.p+size_t(g.li(k,g.pc))*g.nr,
                                g.nr*sizeof(T),hipMemcpyDeviceToHost));
    g.bcast(local.data(),g.nr,owner,g.row);
    for (int i=0;i<g.nr;++i) full[g.rows[i]]=local[i];
    g.sum(full.data(),g.n,g.col);
    return full;
}

template<class T> void store_column(Grid& g, Device<T>& a, int k, const std::vector<T>& v) {
    if (g.c!=(k/g.b)%g.pc) return;
    std::vector<T> local(g.nr);
    for (int i=0;i<g.nr;++i) local[i]=v[g.rows[i]];
    HIP(hipMemcpy(a.p+size_t(g.li(k,g.pc))*g.nr,local.data(),g.nr*sizeof(T),hipMemcpyHostToDevice));
}

struct Phases { double reduction=0, tridiag=0, back=0; };

void tridiagonal_elpa(Grid&, std::vector<double>&, std::vector<double>&, std::vector<double>&);

template<class T> void backtransform(Grid& g, rocblas_handle h, Device<T>& a,
    Device<T>& q, const std::vector<double>& tau, int block) {
    // H_start ... H_end = I - V T V^T. V is row-distributed, replicated over process columns.
    for (int end=g.n-1;end>0;) {
        int start=((end-1)/block)*block, width=end-start;
        Device<T> v(size_t(g.nr)*width), gram(size_t(width)*width), dt(size_t(width)*width);
        Device<T> y(size_t(width)*g.nc), ty(size_t(width)*g.nc);
        std::vector<T> hostv(v.n);
        for (int k=start;k<end;) {
            int take=std::min(end-k,g.b-k%g.b), owner=(k/g.b)%g.pc;
            size_t offset=size_t(k-start)*g.nr, count=size_t(take)*g.nr;
            T* buf=g.gpu_mpi?v.p+offset:hostv.data()+offset;
            if (g.c==owner) HIP(hipMemcpy(buf,a.p+size_t(g.li(k,g.pc))*g.nr,count*sizeof(T),
                g.gpu_mpi?hipMemcpyDeviceToDevice:hipMemcpyDeviceToHost));
            if (g.gpu_mpi) HIP(hipDeviceSynchronize());
            if (g.gpu_mpi) g.device_bcast(buf,count,owner,g.row);
            else g.bcast(buf,count,owner,g.row);
            k+=take;
        }
        if (!g.gpu_mpi) v.put(hostv);
        gemm(h,true,width,width,g.nr,T(1),v.p,g.nr,v.p,g.nr,T(0),gram.p,width);
        g.device_sum(gram,g.col);
        auto gv=gram.get();
        std::vector<double> t(size_t(width)*width,0), x(width);
        for (int i=0;i<width;++i) {
            for (int j=0;j<i;++j) x[j]=-tau[start+i]*gv[j+size_t(i)*width];
            for (int j=0;j<i;++j) for (int l=j;l<i;++l)
                t[j+size_t(i)*width]+=t[j+size_t(l)*width]*x[l];
            t[i+size_t(i)*width]=tau[start+i];
        }
        std::vector<T> tt(t.begin(),t.end()); dt.put(tt);
        bulk_gemm(h,true,width,g.nc,g.nr,T(1),v.p,g.nr,q.p,g.nr,T(0),y.p,width);
        g.device_sum(y,g.col);
        gemm(h,false,width,g.nc,width,T(1),dt.p,width,y.p,width,T(0),ty.p,width);
        bulk_gemm(h,false,g.nr,g.nc,width,T(-1),v.p,g.nr,ty.p,width,T(1),q.p,g.nr);
        end=start;
    }
}

template<class T> Phases custom(Grid& g, rocblas_handle h, Device<T>& a,
    Device<T>& q, std::vector<double>& eval, int back_block, int panel, const std::string& tridiag) {
    Phases times;
    Device<int> dr(g.nr), dc(g.nc); dr.put(g.rows); dc.put(g.cols);
    Device<T> dv(g.n), dw(g.n), vx(g.nc), vy(g.nr), y(g.nc), vr(g.nr);
    Device<T> u(size_t(g.nr)*2*panel), dcross(size_t(g.nc)*2*panel), coeff(2*panel);
    HIP(hipMemset(vy.p,0,vy.n*sizeof(T)));
    std::vector<double> d(g.n), e(g.n,0), tau(g.n-1);
    double start=g.tick();
    for (int k=0;k<g.n-1;++k) {
        if (panel>1) {
            if (k%panel==0) {
                HIP(hipMemset(u.p,0,u.n*sizeof(T))); HIP(hipMemset(dcross.p,0,dcross.n*sizeof(T)));
            } else if (g.c==(k/g.b)%g.pc) {
                correct_column<<<(g.nr+255)/256,256>>>(a.p,u.p,dcross.p,g.nr,g.nc,g.li(k,g.pc),panel);
                HIP(hipGetLastError());
            }
        }
        auto v=column(g,a,k);
        d[k]=v[k];
        double scale=0, norm=0;
        for (int i=k+2;i<g.n;++i) scale=std::max(scale,std::abs(double(v[i])));
        if (scale!=0) {
            for (int i=k+2;i<g.n;++i) { double x=double(v[i])/scale; norm+=x*x; }
            norm=scale*std::sqrt(norm);
        }
        double alpha=v[k+1], beta=alpha;
        if (norm!=0) {
            beta=-std::copysign(std::hypot(alpha,norm),alpha);
            tau[k]=(beta-alpha)/beta;
            for (int i=k+2;i<g.n;++i) v[i]=T(v[i]/(alpha-beta));
        }
        e[k]=beta;
        std::fill(v.begin(),v.begin()+k+1,T(0)); v[k+1]=1;
        if (tau[k]!=0) {
            dv.put(v);
            std::vector<T> vc(g.nc);
            for (int j=0;j<g.nc;++j) vc[j]=v[g.cols[j]];
            vx.put(vc);
            int r0=std::upper_bound(g.rows.begin(),g.rows.end(),k)-g.rows.begin();
            int c0=std::upper_bound(g.cols.begin(),g.cols.end(),k)-g.cols.begin();
            int mr=g.nr-r0, mc=g.nc-c0;
            if (mr && mc) gemv(h,false,mr,mc,g.nr,a.p+r0+size_t(c0)*g.nr,vx.p+c0,vy.p+r0);
            else HIP(hipMemset(vy.p,0,vy.n*sizeof(T)));
            g.device_sum(vy,g.row);
            if (panel>1 && k%panel!=0) {
                gemv(h,true,g.nc,2*panel,g.nc,dcross.p,vx.p,coeff.p);
                g.device_sum(coeff,g.row);
                gemv(h,false,g.nr,2*panel,g.nr,u.p,coeff.p,vy.p,T(-1),T(1));
            }
            auto local=vy.get();
            std::vector<T> w(g.n,0);
            for (int i=0;i<g.nr;++i) if (g.rows[i]>k) w[g.rows[i]]=T(tau[k]*local[i]);
            g.sum(w.data(),g.n,g.col);
            double dot=0;
            for (int i=k+1;i<g.n;++i) dot+=double(v[i])*w[i];
            for (int i=k+1;i<g.n;++i) w[i]-=T(0.5*tau[k]*dot*v[i]);
            dw.put(w);
            if (panel>1) pack_panel<<<(std::max(g.nr,g.nc)+255)/256,256>>>(u.p,dcross.p,dv.p,dw.p,dr.p,dc.p,g.nr,g.nc,k%panel,panel);
            else if (mr && mc) rank2<<<(size_t(mr)*mc+255)/256,256>>>(a.p,dv.p,dw.p,dr.p,dc.p,g.nr,mr,mc,r0,c0);
            HIP(hipGetLastError());
        }
        store_column(g,a,k,v);
        if (panel>1 && ((k+1)%panel==0 || k==g.n-2)) {
            int r0=std::upper_bound(g.rows.begin(),g.rows.end(),k)-g.rows.begin();
            int c0=std::upper_bound(g.cols.begin(),g.cols.end(),k)-g.cols.begin();
            if (r0<g.nr && c0<g.nc) bulk_gemm(h,false,g.nr-r0,g.nc-c0,2*panel,T(-1),
                u.p+r0,g.nr,dcross.p+c0,g.nc,T(1),a.p+r0+size_t(c0)*g.nr,g.nr,true);
        }
    }
    d.back()=column(g,a,g.n-1).back();
    times.reduction=g.tick()-start;
    // Distributed CPU divide-and-conquer. Only O(n) d/e are replicated.
    std::vector<double> z(size_t(g.nr)*g.nc);
    if (tridiag=="elpa") tridiagonal_elpa(g,d,e,z);
    else {
        int one=1, info=0, lw=-1, liw=-1, iwq=0; double wq=0;
        pdstedc_("I",&g.n,d.data(),e.data(),z.data(),&one,&one,g.desc,&wq,&lw,&iwq,&liw,&info);
        if (info) throw std::runtime_error("pdstedc workspace query: "+std::to_string(info));
        lw=int(wq); liw=iwq;
        std::vector<double> work(lw); std::vector<int> iwork(liw);
        pdstedc_("I",&g.n,d.data(),e.data(),z.data(),&one,&one,g.desc,work.data(),&lw,iwork.data(),&liw,&info);
        if (info) throw std::runtime_error("pdstedc: "+std::to_string(info));
    }
    eval=d;
    std::vector<T> zt(z.begin(),z.end()); q.put(zt);
    double middle=g.tick(); times.tridiag=middle-start-times.reduction;
    // Q = H_0 H_1 ... H_(n-2) Z: left multiplication in reverse order.
    if (back_block>1) backtransform(g,h,a,q,tau,back_block);
    else for (int k=g.n-2;k>=0;--k) if (tau[k]!=0) {
        auto v=column(g,a,k); dv.put(v);
        std::vector<T> local(g.nr);
        for (int i=0;i<g.nr;++i) local[i]=v[g.rows[i]];
        vr.put(local);
        gemv(h,true,g.nr,g.nc,g.nr,q.p,vr.p,y.p);
        g.device_sum(y,g.col);
        rank1<<<(size_t(g.nr)*g.nc+255)/256,256>>>(q.p,dv.p,y.p,dr.p,g.nr,g.nc,T(tau[k]));
        HIP(hipGetLastError());
    }
    times.back=g.tick()-middle;
    return times;
}

double entry(int i, int j, int n, const std::string& matrix) {
    if (matrix=="diagonal") return i==j ? double((i*17)%n)/n : 0;
    if (matrix=="clustered") return i==j ? 1.+1e-9*(i%3) : 0;
    if (matrix=="lowrank") {
        double x=0;
        for (int k=1;k<=8;++k) x+=std::sin((i+1)*k*.13)*std::sin((j+1)*k*.13)/n;
        return x+(i==j?1e-7*(1.+double(i)/n):0);
    }
    unsigned long long lo=std::min(i,j)+1, hi=std::max(i,j)+1;
    unsigned long long x=lo*0x9e3779b97f4a7c15ULL+hi*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; x^=x>>31;
    double val=(double(x>>11)*0x1.0p-53-.5)/std::sqrt(double(n));
    if (matrix=="indefinite") return val;
    if (matrix!="random") throw std::runtime_error("unknown matrix family");
    return val+(i==j ? 3.+double(i)/n : 0);
}

double quantize(double x, const std::string& quant) {
    if (quant=="fp16") return float(__float2half(float(x)));
    if (quant=="bf16") {
        float f=x; unsigned u; std::memcpy(&u,&f,4);
        u+=0x7fff+((u>>16)&1); u&=0xffff0000; std::memcpy(&f,&u,4); return f;
    }
    if (quant!="none") throw std::runtime_error("quant must be none, fp16, or bf16");
    return x;
}

std::vector<double> matrix(Grid& g, const std::string& name, const std::string& quant) {
    std::vector<double> a(size_t(g.nr)*g.nc);
    for (int j=0;j<g.nc;++j) for (int i=0;i<g.nr;++i)
        a[i+size_t(j)*g.nr]=quantize(entry(g.rows[i],g.cols[j],g.n,name),quant);
    return a;
}

// Exact distributed GPU SUMMA validation. Host panel staging is outside solver timings.
std::vector<double> product(Grid& g, rocblas_handle h,
    const std::vector<double>& a, const std::vector<double>& b) {
    Device<double> da(size_t(g.nr)*g.b), db(size_t(g.b)*g.nc), out(size_t(g.nr)*g.nc);
    HIP(hipMemset(out.p,0,out.n*sizeof(double)));
    for (int k=0;k<g.n;k+=g.b) {
        int width=std::min(g.b,g.n-k);
        std::vector<double> ap(size_t(g.nr)*width), bp(size_t(width)*g.nc);
        if (g.c==(k/g.b)%g.pc)
            std::copy_n(a.data()+size_t(g.li(k,g.pc))*g.nr,ap.size(),ap.data());
        if (g.r==(k/g.b)%g.pr) for (int j=0;j<g.nc;++j)
            std::copy_n(b.data()+g.li(k,g.pr)+size_t(j)*g.nr,width,bp.data()+size_t(j)*width);
        g.bcast(ap.data(),ap.size(),(k/g.b)%g.pc,g.row);
        g.bcast(bp.data(),bp.size(),(k/g.b)%g.pr,g.col);
        HIP(hipMemcpy(da.p,ap.data(),ap.size()*sizeof(double),hipMemcpyHostToDevice));
        HIP(hipMemcpy(db.p,bp.data(),bp.size()*sizeof(double),hipMemcpyHostToDevice));
        double one=1;
        BLAS(rocblas_dgemm(h,rocblas_operation_none,rocblas_operation_none,g.nr,g.nc,width,
                          &one,da.p,g.nr,db.p,width,&one,out.p,g.nr));
    }
    return out.get();
}

void elpa_check(int error) { if (error!=ELPA_OK) throw std::runtime_error(elpa_strerr(error)); }
elpa_t elpa_handle(Grid& g, int solver) {
    elpa_check(elpa_init(ELPA_API_VERSION));
    int error; elpa_t e=elpa_allocate(&error); elpa_check(error);
    auto set=[&](const char* k,int v) { elpa_set_integer(e,k,v,&error); elpa_check(error); };
    set("na",g.n); set("nev",g.n); set("local_nrows",g.nr); set("local_ncols",g.nc);
    set("nblk",g.b); set("mpi_comm_parent",int(MPI_Comm_c2f(MPI_COMM_WORLD)));
    set("process_row",g.r); set("process_col",g.c); elpa_check(elpa_setup(e));
    set("solver",solver); set("amd-gpu",1); set("use_gpu_id",0);
    elpa_check(elpa_setup_gpu(e));
    return e;
}
void elpa_release(elpa_t e) {
    int error; elpa_deallocate(e,&error); elpa_check(error); elpa_uninit(&error); elpa_check(error);
}
void tridiagonal_elpa(Grid& g, std::vector<double>& d, std::vector<double>& e, std::vector<double>& z) {
    auto handle=elpa_handle(g,ELPA_SOLVER_1STAGE);
    int error;
    // This public component accepts HOST arrays and stages its distributed GPU work internally.
    elpa_solve_tridiagonal_d(handle,d.data(),e.data(),z.data(),&error);
    elpa_check(error); elpa_release(handle);
}
template<class T> void elpa_solve(Grid& g, Device<T>& a, Device<T>& q,
                                  std::vector<double>& eigenvalues, int solver) {
    auto e=elpa_handle(g,solver); int error;
    Device<T> dd(g.n);
    if constexpr(std::is_same_v<T,double>) elpa_eigenvectors_double(e,a.p,dd.p,q.p,&error);
    else elpa_eigenvectors_float(e,a.p,dd.p,q.p,&error);
    elpa_check(error); auto d=dd.get(); eigenvalues.assign(d.begin(),d.end());
    elpa_release(e);
}

template<class T> void run(Grid& g, const std::string& solver, const std::string& name,
    const std::string& quant, bool reference, int repeat, int back_block, int panel, const std::string& tridiag) {
    if (repeat<1 || back_block<1 || panel<1) throw std::runtime_error("repeat and block widths must be positive");
    if (tridiag!="scalapack" && tridiag!="elpa") throw std::runtime_error("tridiag must be scalapack or elpa");
    rocblas_handle h; BLAS(rocblas_create_handle(&h));
    double begin=g.tick();
    auto original=matrix(g,name,"none"), input=matrix(g,name,quant);
    std::vector<T> at(input.begin(),input.end());
    Device<T> a(at.size()), q(at.size()); a.put(at);
    double setup=g.tick()-begin;
    std::vector<double> eval(g.n);
    Phases phases;
    double total=0;
    for (int trial=0;trial<repeat;++trial) {
        a.put(at); g.comm=0; double start=g.tick();
        if (solver=="custom") phases=custom(g,h,a,q,eval,back_block,panel,tridiag);
        else if (solver=="elpa1" || solver=="elpa2")
            elpa_solve(g,a,q,eval,solver=="elpa1"?ELPA_SOLVER_1STAGE:ELPA_SOLVER_2STAGE);
        else if (solver=="rocsolver") {
            if (g.pr*g.pc!=1) throw std::runtime_error("rocSOLVER reference requires one rank");
            Device<T> d(g.n), e(g.n); Device<int> info(1);
            if constexpr(std::is_same_v<T,double>) BLAS(rocsolver_dsyevd(h,rocblas_evect_original,
                rocblas_fill_lower,g.n,a.p,g.nr,d.p,e.p,info.p));
            else BLAS(rocsolver_ssyevd(h,rocblas_evect_original,
                rocblas_fill_lower,g.n,a.p,g.nr,d.p,e.p,info.p));
            if (info.get()[0]) throw std::runtime_error("rocsolver info!=0");
            auto dd=d.get(); eval.assign(dd.begin(),dd.end());
            HIP(hipMemcpy(q.p,a.p,a.n*sizeof(T),hipMemcpyDeviceToDevice));
        } else throw std::runtime_error("unknown solver");
        total=g.maximum(g.tick()-start);
        if (g.rank==0) std::printf("TIMING solver=%s trial=%d n=%d grid=%dx%d block=%d precision=%s matrix=%s quant=%s total=%.9g reduction=%.9g tridiag=%.9g back=%.9g back_block=%d gpu_mpi=%d panel=%d tridiag_backend=%s rccl=%d half_bulk=%d\n",
            solver.c_str(),trial,g.n,g.pr,g.pc,g.b,std::is_same_v<T,double>?"fp64":"fp32",name.c_str(),quant.c_str(),total,
            phases.reduction,phases.tridiag,phases.back,back_block,int(g.gpu_mpi),panel,tridiag.c_str(),int(g.rccl),int(half_bulk));
    }
    double comm=g.maximum(g.comm); g.comm=0;
    auto qt=q.get(); std::vector<double> qd(qt.begin(),qt.end());
    auto aq=product(g,h,original,qd);
    auto aiq=quant=="none" ? aq : product(g,h,input,qd);
    std::vector<double> trans(qd.size());
    int one=1; double alpha=1, beta=0;
    pdtran_(&g.n,&g.n,&alpha,qd.data(),&one,&one,g.desc,&beta,trans.data(),&one,&one,g.desc);
    auto gram=product(g,h,trans,qd);
    double sums[5]={};
    for (int j=0;j<g.nc;++j) for (int i=0;i<g.nr;++i) {
        size_t z=i+size_t(j)*g.nr;
        double diff=aq[z]-qd[z]*eval[g.cols[j]];
        double orth=gram[z]-(g.rows[i]==g.cols[j]?1.:0.);
        sums[0]+=diff*diff; sums[1]+=original[z]*original[z]; sums[2]+=orth*orth;
        sums[3]+=(input[z]-original[z])*(input[z]-original[z]);
        double di=aiq[z]-qd[z]*eval[g.cols[j]];
        sums[4]+=di*di;
    }
    g.sum(sums,5,MPI_COMM_WORLD);
    double eigerr=-1, smallerr=-1;
    if (g.n<=512) {
        std::vector<double> ref(g.n);
        if (!g.rank) {
            std::vector<double> dense(size_t(g.n)*g.n);
            for (int j=0;j<g.n;++j) for (int i=0;i<g.n;++i) dense[i+size_t(j)*g.n]=entry(i,j,g.n,name);
            int info,lwork=std::max(1,3*g.n); std::vector<double> work(lwork);
            dsyev_("N","L",&g.n,dense.data(),&g.n,ref.data(),work.data(),&lwork,&info);
            if (info) throw std::runtime_error("dsyev reference failed");
        }
        g.bcast(ref.data(),g.n,0,MPI_COMM_WORLD);
        double diff=0,scale=0;
        for(int i=0;i<g.n;++i) { diff=std::max(diff,std::abs(ref[i]-eval[i])); scale=std::max(scale,std::abs(ref[i])); }
        eigerr=diff/scale;
        if (ref.front()!=0) smallerr=std::abs(ref.front()-eval.front())/std::abs(ref.front());
    } else if (reference) {
        Device<double> ar(original.size()), qr(original.size()); ar.put(original);
        std::vector<double> ref(g.n); elpa_solve(g,ar,qr,ref,ELPA_SOLVER_2STAGE);
        double diff=0,scale=0;
        for(int i=0;i<g.n;++i) { diff=std::max(diff,std::abs(ref[i]-eval[i])); scale=std::max(scale,std::abs(ref[i])); }
        eigerr=diff/scale;
        if (ref.front()!=0) smallerr=std::abs(ref.front()-eval.front())/std::abs(ref.front());
    }
    double residual=std::sqrt(sums[0]/sums[1]), orth=std::sqrt(sums[2]);
    double input_residual=std::sqrt(sums[4]/sums[1]);
    int negative=std::count_if(eval.begin(),eval.end(),[](double x){return x<0;});
    if (!g.rank) std::printf("RESULT solver=%s n=%d grid=%dx%d block=%d precision=%s matrix=%s quant=%s setup=%.9g total=%.9g comm=%.9g residual=%.9g orthogonality=%.9g eig_rel_max=%.9g input_rel=%.9g eig_min=%.9g eig_max=%.9g input_residual=%.9g eig_small_rel=%.9g negative=%d\n",
        solver.c_str(),g.n,g.pr,g.pc,g.b,std::is_same_v<T,double>?"fp64":"fp32",name.c_str(),quant.c_str(),setup,total,comm,
        residual,orth,eigerr,std::sqrt(sums[3]/sums[1]),eval.front(),eval.back(),input_residual,smallerr,negative);
    BLAS(rocblas_destroy_handle(h));
    double tol=std::is_same_v<T,double>?1e-10:1e-4;
    if (!std::isfinite(residual) || !std::isfinite(input_residual) || input_residual>tol || !std::isfinite(orth) || orth>tol*std::sqrt(g.n))
        throw std::runtime_error("numerical validation failed");
}

int main(int argc,char** argv) {
    // HIP must initialize before MPI on this Frontier software stack.
    HIP(hipSetDevice(0)); HIP(hipFree(nullptr));
    MPI(MPI_Init(&argc,&argv));
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        auto arg=[&](const char* key,const char* def) { for(int i=1;i<argc-1;++i) if(!std::strcmp(argv[i],key)) return std::string(argv[i+1]); return std::string(def); };
        int n=std::stoi(arg("--n","64")), b=std::stoi(arg("--block","16"));
        int pr=std::stoi(arg("--pr","1")), pc=std::stoi(arg("--pc","1"));
        std::string precision=arg("--precision","fp64");
        half_bulk=std::stoi(arg("--half-bulk","0"));
        if (half_bulk<0 || half_bulk>2) throw std::runtime_error("half bulk must be 0, 1, or 2");
        if (half_bulk && precision!="fp32") throw std::runtime_error("half bulk experiment requires FP32 state");
        {
            Grid g(n,b,pr,pc,arg("--gpu-mpi","0")=="1",arg("--rccl","0")=="1");
            if (precision=="fp64") run<double>(g,arg("--solver","custom"),arg("--matrix","random"),arg("--quant","none"),arg("--reference","0")=="1",std::stoi(arg("--repeat","1")),std::stoi(arg("--back-block","1")),std::stoi(arg("--panel","1")),arg("--tridiag","scalapack"));
            else if (precision=="fp32") run<float>(g,arg("--solver","custom"),arg("--matrix","random"),arg("--quant","none"),arg("--reference","0")=="1",std::stoi(arg("--repeat","1")),std::stoi(arg("--back-block","1")),std::stoi(arg("--panel","1")),arg("--tridiag","scalapack"));
            else throw std::runtime_error("precision must be fp64 or fp32");
        }
        MPI(MPI_Finalize()); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what()); MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
