// Standalone research solver: real symmetric, full eigenvectors, 2D block cyclic.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
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
    Grid(int n_, int b_, int pr_, int pc_, bool gm): n(n_), b(b_), pr(pr_), pc(pc_), gpu_mpi(gm) {
        MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        int size; MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
        if (pr*pc != size || n < 2 || b < 1 || pr < 1 || pc < 1)
            throw std::runtime_error("invalid size, block, or grid");
        r = rank/pc; c = rank%pc;
        MPI(MPI_Comm_split(MPI_COMM_WORLD, r, c, &row));
        MPI(MPI_Comm_split(MPI_COMM_WORLD, c, r, &col));
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
    ~Grid() { Cblacs_gridexit(ctxt); MPI_Comm_free(&row); MPI_Comm_free(&col); }
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
        if (gpu_mpi) sum(d.p, d.n, comm_);
        else { auto h=d.get(); sum(h.data(), h.size(), comm_); d.put(h); }
    }
    double tick() { HIP(hipDeviceSynchronize()); MPI(MPI_Barrier(MPI_COMM_WORLD)); return MPI_Wtime(); }
    double maximum(double x) { double y; MPI(MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD)); return y; }
};

template<class T> void gemv(rocblas_handle h, bool trans, int m, int n,
                            T* a, T* x, T* y) {
    T one=1, zero=0;
    auto op=trans?rocblas_operation_transpose:rocblas_operation_none;
    if constexpr(std::is_same_v<T,double>) BLAS(rocblas_dgemv(h,op,m,n,&one,a,m,x,1,&zero,y,1));
    else BLAS(rocblas_sgemv(h,op,m,n,&one,a,m,x,1,&zero,y,1));
}

template<class T> __global__ void rank2(T* a, const T* v, const T* w,
    const int* rows, const int* cols, int nr, int nc, int k) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z>=size_t(nr)*nc) return;
    int i=rows[z%nr], j=cols[z/nr];
    if (i>k && j>k) a[z] -= v[i]*w[j] + w[i]*v[j];
}

template<class T> __global__ void rank1(T* q, const T* v, const T* y,
    const int* rows, int nr, int nc, T tau) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (z<size_t(nr)*nc) q[z] -= tau*v[rows[z%nr]]*y[z/nr];
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

template<class T> Phases custom(Grid& g, rocblas_handle h, Device<T>& a,
                               Device<T>& q, std::vector<double>& eval) {
    Phases times;
    Device<int> dr(g.nr), dc(g.nc); dr.put(g.rows); dc.put(g.cols);
    Device<T> dv(g.n), dw(g.n), vx(g.nc), vy(g.nr), y(g.nc), vr(g.nr);
    std::vector<double> d(g.n), e(g.n,0), tau(g.n-1);
    double start=g.tick();
    for (int k=0;k<g.n-1;++k) {
        auto v=column(g,a,k);
        d[k]=v[k];
        double norm=0;
        for (int i=k+2;i<g.n;++i) norm=std::hypot(norm,double(v[i]));
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
            gemv(h,false,g.nr,g.nc,a.p,vx.p,vy.p);
            g.device_sum(vy,g.row);
            auto local=vy.get();
            std::vector<T> w(g.n,0);
            for (int i=0;i<g.nr;++i) if (g.rows[i]>k) w[g.rows[i]]=T(tau[k]*local[i]);
            g.sum(w.data(),g.n,g.col);
            double dot=0;
            for (int i=k+1;i<g.n;++i) dot+=double(v[i])*w[i];
            for (int i=k+1;i<g.n;++i) w[i]-=T(0.5*tau[k]*dot*v[i]);
            dw.put(w);
            rank2<<<(size_t(g.nr)*g.nc+255)/256,256>>>(a.p,dv.p,dw.p,dr.p,dc.p,g.nr,g.nc,k);
            HIP(hipGetLastError());
        }
        store_column(g,a,k,v);
    }
    d.back()=column(g,a,g.n-1).back();
    times.reduction=g.tick()-start;
    // Distributed CPU divide-and-conquer. Only O(n) d/e are replicated.
    std::vector<double> z(size_t(g.nr)*g.nc);
    int one=1, info=0, lw=-1, liw=-1, iwq=0; double wq=0;
    pdstedc_("I",&g.n,d.data(),e.data(),z.data(),&one,&one,g.desc,&wq,&lw,&iwq,&liw,&info);
    if (info) throw std::runtime_error("pdstedc workspace query: "+std::to_string(info));
    lw=int(wq); liw=iwq;
    std::vector<double> work(lw); std::vector<int> iwork(liw);
    pdstedc_("I",&g.n,d.data(),e.data(),z.data(),&one,&one,g.desc,work.data(),&lw,iwork.data(),&liw,&info);
    if (info) throw std::runtime_error("pdstedc: "+std::to_string(info));
    eval=d;
    std::vector<T> zt(z.begin(),z.end()); q.put(zt);
    double middle=g.tick(); times.tridiag=middle-start-times.reduction;
    // Q = H_0 H_1 ... H_(n-2) Z: left multiplication in reverse order.
    for (int k=g.n-2;k>=0;--k) if (tau[k]!=0) {
        auto v=column(g,a,k); dv.put(v);
        std::vector<T> local(g.nr);
        for (int i=0;i<g.nr;++i) local[i]=v[g.rows[i]];
        vr.put(local);
        gemv(h,true,g.nr,g.nc,q.p,vr.p,y.p);
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
    return val+(i==j ? 3.+double(i)/n : 0);
}

double quantize(double x, const std::string& quant) {
    if (quant=="fp16") return float(__float2half(float(x)));
    if (quant=="bf16") {
        float f=x; unsigned u; std::memcpy(&u,&f,4);
        u+=0x7fff+((u>>16)&1); u&=0xffff0000; std::memcpy(&f,&u,4); return f;
    }
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
template<class T> void elpa_solve(Grid& g, Device<T>& a, Device<T>& q,
                                  std::vector<double>& eigenvalues, int solver) {
    elpa_check(elpa_init(ELPA_API_VERSION));
    int error; elpa_t e=elpa_allocate(&error); elpa_check(error);
    auto set=[&](const char* k,int v) { elpa_set_integer(e,k,v,&error); elpa_check(error); };
    set("na",g.n); set("nev",g.n); set("local_nrows",g.nr); set("local_ncols",g.nc);
    set("nblk",g.b); set("mpi_comm_parent",int(MPI_Comm_c2f(MPI_COMM_WORLD)));
    set("process_row",g.r); set("process_col",g.c); elpa_check(elpa_setup(e));
    set("solver",solver); set("amd-gpu",1); set("use_gpu_id",0);
    elpa_check(elpa_setup_gpu(e));
    Device<T> dd(g.n);
    if constexpr(std::is_same_v<T,double>) elpa_eigenvectors_double(e,a.p,dd.p,q.p,&error);
    else elpa_eigenvectors_float(e,a.p,dd.p,q.p,&error);
    elpa_check(error); auto d=dd.get(); eigenvalues.assign(d.begin(),d.end());
    elpa_deallocate(e,&error); elpa_check(error); elpa_uninit(&error); elpa_check(error);
}

template<class T> void run(Grid& g, const std::string& solver, const std::string& name,
    const std::string& quant, bool reference, int repeat) {
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
        a.put(at); double start=g.tick();
        if (solver=="custom") phases=custom(g,h,a,q,eval);
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
        if (g.rank==0) std::printf("TIMING solver=%s trial=%d n=%d grid=%dx%d block=%d precision=%s matrix=%s quant=%s total=%.9g reduction=%.9g tridiag=%.9g back=%.9g\n",
            solver.c_str(),trial,g.n,g.pr,g.pc,g.b,std::is_same_v<T,double>?"fp64":"fp32",name.c_str(),quant.c_str(),total,
            phases.reduction,phases.tridiag,phases.back);
    }
    double comm=g.maximum(g.comm); g.comm=0;
    auto qt=q.get(); std::vector<double> qd(qt.begin(),qt.end());
    auto aq=product(g,h,original,qd);
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
        double di=aq[z]-qd[z]*eval[g.cols[j]];
        (void)di;
    }
    g.sum(sums,5,MPI_COMM_WORLD);
    double eigerr=-1;
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
    } else if (reference) {
        Device<double> ar(original.size()), qr(original.size()); ar.put(original);
        std::vector<double> ref(g.n); elpa_solve(g,ar,qr,ref,ELPA_SOLVER_2STAGE);
        double diff=0,scale=0;
        for(int i=0;i<g.n;++i) { diff=std::max(diff,std::abs(ref[i]-eval[i])); scale=std::max(scale,std::abs(ref[i])); }
        eigerr=diff/scale;
    }
    double residual=std::sqrt(sums[0]/sums[1]), orth=std::sqrt(sums[2]);
    if (!g.rank) std::printf("RESULT solver=%s n=%d grid=%dx%d block=%d precision=%s matrix=%s quant=%s setup=%.9g total=%.9g comm=%.9g residual=%.9g orthogonality=%.9g eig_rel_max=%.9g input_rel=%.9g eig_min=%.9g eig_max=%.9g\n",
        solver.c_str(),g.n,g.pr,g.pc,g.b,std::is_same_v<T,double>?"fp64":"fp32",name.c_str(),quant.c_str(),setup,total,comm,
        residual,orth,eigerr,std::sqrt(sums[3]/sums[1]),eval.front(),eval.back());
    BLAS(rocblas_destroy_handle(h));
    double tol=std::is_same_v<T,double>?1e-10:1e-4;
    if (quant=="fp16") tol=2e-3;
    if (quant=="bf16") tol=2e-2;
    if (!std::isfinite(residual) || residual>tol || !std::isfinite(orth) || orth>tol*std::sqrt(g.n))
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
        {
            Grid g(n,b,pr,pc,arg("--gpu-mpi","0")=="1");
            if (precision=="fp64") run<double>(g,arg("--solver","custom"),arg("--matrix","random"),arg("--quant","none"),arg("--reference","0")=="1",std::stoi(arg("--repeat","1")));
            else if (precision=="fp32") run<float>(g,arg("--solver","custom"),arg("--matrix","random"),arg("--quant","none"),arg("--reference","0")=="1",std::stoi(arg("--repeat","1")));
            else throw std::runtime_error("precision must be fp64 or fp32");
        }
        MPI(MPI_Finalize()); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what()); MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
