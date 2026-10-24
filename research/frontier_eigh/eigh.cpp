// Real symmetric eigensolver: HIP Householder reduction, ELPA tridiagonal solve,
// and HIP compact-WY backtransform. Matrices stay 2D block-cyclic distributed.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rccl/rccl.h>
#include <elpa/elpa.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
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
void pdtran_(const int*, const int*, const double*, const double*, const int*,
             const int*, const int*, const double*, double*, const int*, const int*, const int*);
void pdgemr2d_(const int*, const int*, const double*, const int*, const int*, const int*,
               double*, const int*, const int*, const int*, const int*);
void psgemr2d_(const int*, const int*, const float*, const int*, const int*, const int*,
               float*, const int*, const int*, const int*, const int*);
void dsyev_(const char*, const char*, const int*, double*, const int*, double*,
            double*, const int*, int*);
void dtrmv_(const char*, const char*, const char*, const int*, const double*,
            const int*, double*, const int*);
}

#define HIP(x) do { auto s=(x); if (s!=hipSuccess) throw std::runtime_error( \
    std::string(#x)+": "+hipGetErrorString(s)); } while (0)
#define BLAS(x) do { auto s=(x); if (s!=rocblas_status_success) throw std::runtime_error( \
    std::string(#x)+": "+std::to_string(s)); } while (0)
#define MPI(x) do { if ((x)!=MPI_SUCCESS) throw std::runtime_error(#x); } while (0)
#define CCL(x) do { auto s=(x); if (s!=ncclSuccess) throw std::runtime_error( \
    std::string(#x)+": "+ncclGetErrorString(s)); } while (0)

template<class T> struct Device {
    T* p=nullptr;
    size_t n;
    explicit Device(size_t count): n(count) { HIP(hipMalloc(&p,n*sizeof(T))); }
    ~Device() { hipFree(p); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
    void put(const std::vector<T>& h) { HIP(hipMemcpy(p,h.data(),n*sizeof(T),hipMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        HIP(hipMemcpy(h.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));
        return h;
    }
};

struct Grid {
    int n, b, pr, pc, rank, r, c, nr, nc, ctxt, desc[9];
    MPI_Comm row, col;
    ncclComm_t ccl_row, ccl_col;
    std::vector<int> rows, cols;

    Grid(int n_, int b_, int pr_, int pc_): n(n_), b(b_), pr(pr_), pc(pc_) {
        int size;
        MPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
        MPI(MPI_Comm_size(MPI_COMM_WORLD,&size));
        if (n<2 || b<1 || pr<1 || pc<1 || pr*pc!=size)
            throw std::runtime_error("invalid matrix size, block size, or process grid");
        r=rank/pc; c=rank%pc;
        MPI(MPI_Comm_split(MPI_COMM_WORLD,r,c,&row));
        MPI(MPI_Comm_split(MPI_COMM_WORLD,c,r,&col));
        auto init=[](MPI_Comm comm, int rank, int size, ncclComm_t* out) {
            ncclUniqueId id;
            if (!rank) CCL(ncclGetUniqueId(&id));
            MPI(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,comm));
            CCL(ncclCommInitRank(out,size,id,rank));
        };
        init(row,c,pc,&ccl_row); init(col,r,pr,&ccl_col);
        for (int i=0;i<n;++i) {
            if ((i/b)%pr==r) rows.push_back(i);
            if ((i/b)%pc==c) cols.push_back(i);
        }
        nr=rows.size(); nc=cols.size();
        if (!nr || !nc) throw std::runtime_error("every rank must own rows and columns");
        Cblacs_get(-1,0,&ctxt); Cblacs_gridinit(&ctxt,"R",pr,pc);
        int zero=0, info;
        descinit_(desc,&n,&n,&b,&b,&zero,&zero,&ctxt,&nr,&info);
        if (info) throw std::runtime_error("descinit failed");
    }
    ~Grid() {
        ncclCommDestroy(ccl_row); ncclCommDestroy(ccl_col);
        Cblacs_gridexit(ctxt); MPI_Comm_free(&row); MPI_Comm_free(&col);
    }
    int li(int i, int p) const { return (i/b/p)*b+i%b; }
    template<class T> static MPI_Datatype type() { return std::is_same_v<T,double>?MPI_DOUBLE:MPI_FLOAT; }
    template<class T> static ncclDataType_t device_type() { return std::is_same_v<T,double>?ncclFloat64:ncclFloat32; }
    template<class T> void sum(T* p, int count, MPI_Comm comm) {
        MPI(MPI_Allreduce(MPI_IN_PLACE,p,count,type<T>(),MPI_SUM,comm));
    }
    template<class T> void bcast(T* p, int count, int root, MPI_Comm comm) {
        MPI(MPI_Bcast(p,count,type<T>(),root,comm));
    }
    template<class T> void device_sum(Device<T>& d, MPI_Comm comm) {
        if ((comm==row?pc:pr)>1)
            CCL(ncclAllReduce(d.p,d.p,d.n,device_type<T>(),ncclSum,comm==row?ccl_row:ccl_col,nullptr));
    }
    double tick() { HIP(hipDeviceSynchronize()); MPI(MPI_Barrier(MPI_COMM_WORLD)); return MPI_Wtime(); }
    double maximum(double x) {
        double y; MPI(MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD)); return y;
    }
};

template<class T> void gemv(rocblas_handle h, int m, int n, int lda, T* a, T* x, T* y) {
    T one=1, zero=0;
    if constexpr(std::is_same_v<T,double>)
        BLAS(rocblas_dgemv(h,rocblas_operation_none,m,n,&one,a,lda,x,1,&zero,y,1));
    else BLAS(rocblas_sgemv(h,rocblas_operation_none,m,n,&one,a,lda,x,1,&zero,y,1));
}

template<class T> void gemm(rocblas_handle h, bool trans, int m, int n, int k,
    T alpha, T* a, int lda, T* b, int ldb, T beta, T* c, int ldc) {
    auto op=trans?rocblas_operation_transpose:rocblas_operation_none;
    if constexpr(std::is_same_v<T,double>)
        BLAS(rocblas_dgemm(h,op,rocblas_operation_none,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc));
    else BLAS(rocblas_sgemm(h,op,rocblas_operation_none,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc));
}

template<class T> std::vector<T> compact_t(const std::vector<T>& gram,
    const std::vector<double>& tau, int start, int width) {
    std::vector<double> t(size_t(width)*width,0), x(width);
    int one=1;
    for (int i=0;i<width;++i) {
        for (int j=0;j<i;++j) x[j]=-tau[start+i]*gram[j+size_t(i)*width];
        if (i) dtrmv_("U","N","N",&i,t.data(),&width,x.data(),&one);
        std::copy_n(x.data(),i,t.data()+size_t(i)*width);
        t[i+size_t(i)*width]=tau[start+i];
    }
    return std::vector<T>(t.begin(),t.end());
}

template<class T> void backtransform(Grid& g, rocblas_handle h, Device<T>& a,
    Device<T>& q, const std::vector<double>& tau) {
    // Apply H_start ... H_end = I - V T V^T, in reverse block order.
    constexpr int block=256;
    for (int end=g.n-1;end>0;) {
        int start=((end-1)/block)*block, width=end-start;
        Device<T> v(size_t(g.nr)*width), gram(size_t(width)*width), dt(gram.n);
        Device<T> y(size_t(width)*g.nc), ty(y.n);
        for (int k=start;k<end;) {
            int take=std::min(end-k,g.b-k%g.b), owner=(k/g.b)%g.pc;
            T* buf=v.p+size_t(k-start)*g.nr;
            size_t count=size_t(take)*g.nr;
            if (g.c==owner)
                HIP(hipMemcpyAsync(buf,a.p+size_t(g.li(k,g.pc))*g.nr,count*sizeof(T),hipMemcpyDeviceToDevice));
            CCL(ncclBroadcast(buf,buf,count,Grid::device_type<T>(),owner,g.ccl_row,nullptr));
            k+=take;
        }
        gemm(h,true,width,width,g.nr,T(1),v.p,g.nr,v.p,g.nr,T(0),gram.p,width);
        g.device_sum(gram,g.col);
        dt.put(compact_t(gram.get(),tau,start,width));
        gemm(h,true,width,g.nc,g.nr,T(1),v.p,g.nr,q.p,g.nr,T(0),y.p,width);
        g.device_sum(y,g.col);
        gemm(h,false,width,g.nc,width,T(1),dt.p,width,y.p,width,T(0),ty.p,width);
        gemm(h,false,g.nr,g.nc,width,T(-1),v.p,g.nr,ty.p,width,T(1),q.p,g.nr);
        end=start;
    }
}

__device__ void write_reflector(double norm, double alpha, double diagonal, double* s,
    double* d, double* e, double* tau, int k) {
    double beta=norm!=0?-copysign(hypot(alpha,norm),alpha):alpha;
    d[k]=diagonal; e[k]=beta; tau[k]=norm!=0?(beta-alpha)/beta:0;
    s[5]=alpha-beta;
}

// Scaled norm pairs avoid underflow/overflow in Householder scalar formation.
template<class T> __global__ void local_statistics(const T* v, int nr, int tail,
    int diagonal, int alpha, double* s, double* d, double* e, double* tau, int k, bool single) {
    __shared__ double partial[256];
    __shared__ double scale;
    int t=threadIdx.x;
    double x=0;
    for (int i=tail+t;i<nr;i+=256) x=fmax(x,fabs(double(v[i])));
    partial[t]=x; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (t<stride) partial[t]=fmax(partial[t],partial[t+stride]);
        __syncthreads();
    }
    if (!t) scale=partial[0];
    __syncthreads();
    x=0;
    if (scale!=0) for (int i=tail+t;i<nr;i+=256) { double y=double(v[i])/scale; x+=y*y; }
    partial[t]=x; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (t<stride) partial[t]+=partial[t+stride];
        __syncthreads();
    }
    if (!t) {
        s[0]=scale; s[1]=partial[0];
        s[2]=alpha<0?0:double(v[alpha]);
        s[3]=diagonal<0?0:double(v[diagonal]);
        if (single) write_reflector(scale*sqrt(partial[0]),s[2],s[3],s,d,e,tau,k);
    }
}

__global__ void reflector_scalar(const double* gathered, int pr, double* s,
    double* d, double* e, double* tau, int k) {
    double scale=0, sum=0, alpha=0, diagonal=0;
    for (int r=0;r<pr;++r) scale=fmax(scale,gathered[4*r]);
    for (int r=0;r<pr;++r) {
        double ratio=scale!=0?gathered[4*r]/scale:0;
        sum+=gathered[4*r+1]*ratio*ratio;
        alpha+=gathered[4*r+2]; diagonal+=gathered[4*r+3];
    }
    write_reflector(scale*sqrt(sum),alpha,diagonal,s,d,e,tau,k);
}

template<class T> __global__ void vector_dot(const T* v, T* w, int nr, int start,
    int k, double* s, const double* tau) {
    __shared__ double partial[256];
    int t=threadIdx.x;
    double dot=0;
    for (int i=start+t;i<nr;i+=256) {
        w[i]=T(tau[k]*double(w[i])); dot+=double(v[i])*double(w[i]);
    }
    partial[t]=dot; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (t<stride) partial[t]+=partial[t+stride];
        __syncthreads();
    }
    if (!t) s[4]=partial[0];
}

template<class T> __global__ void form_vectors(T* v, T* vc, T* w, T* column,
    const T* raw, const T* packed, const int* rows, const int* cols,
    int nr, int nc, int padded, int b, int pr, int k, const double* s, const double* tau) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<padded) {
        w[i]=0;
        if (i>=nr || rows[i]<=k) v[i]=0;
        else if (rows[i]==k+1) v[i]=1;
        else v[i]=tau[k]!=0?T(double(raw[i])/s[5]):raw[i];
        if (column && i<nr) column[i]=v[i];
    }
    if (i<nc) {
        int r=cols[i];
        T value=packed[size_t((r/b)%pr)*padded+(r/b/pr)*b+r%b];
        vc[i]=r<=k?T(0):r==k+1?T(1):tau[k]!=0?T(double(value)/s[5]):value;
    }
}

template<class T> __global__ void correct_vectors(T* w, T* wc, const T* input,
    const T* v, const T* vc, const T* packed, const int* cols,
    int nc, int padded, int b, int pr, const double* s, const double* tau, int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<padded) w[i]=input[i]-T(.5*tau[k]*s[4]*double(v[i]));
    if (i<nc) {
        int r=cols[i];
        T value=packed[size_t((r/b)%pr)*padded+(r/b/pr)*b+r%b];
        wc[i]=value-T(.5*tau[k]*s[4]*double(vc[i]));
    }
}

template<class T> __global__ void local_rank2(T* a, const T* v, const T* w,
    const T* vc, const T* wc, int nr, int nc, int r0, int c0) {
    int i=r0+blockIdx.x*blockDim.x+threadIdx.x;
    int j=c0+blockIdx.y*blockDim.y+threadIdx.y;
    if (i<nr && j<nc) a[i+size_t(j)*nr]-=v[i]*wc[j]+w[i]*vc[j];
}

template<class T> void reduce(Grid& g, rocblas_handle h, Device<T>& a,
    std::vector<double>& d, std::vector<double>& e, std::vector<double>& tau) {
    Device<int> rows(g.nr), cols(g.nc); rows.put(g.rows); cols.put(g.cols);
    int padded=((g.n+g.b*g.pr-1)/(g.b*g.pr))*g.b;
    Device<T> v(padded), w(padded), vc(g.nc), wc(g.nc), packed(size_t(padded)*g.pr);
    Device<double> scalars(6), statistics(4*g.pr), dd(g.n), ee(g.n), tt(g.n-1);
    HIP(hipMemsetAsync(dd.p,0,dd.n*sizeof(double)));
    HIP(hipMemsetAsync(ee.p,0,ee.n*sizeof(double)));
    auto gather=[&](Device<T>& local) {
        CCL(ncclAllGather(local.p,packed.p,padded,Grid::device_type<T>(),g.ccl_col,nullptr));
    };
    for (int k=0;k<g.n-1;++k) {
        int owner=(k/g.b)%g.pc;
        T* column=g.c==owner?a.p+size_t(g.li(k,g.pc))*g.nr:nullptr;
        T* raw=g.pr==1?packed.p:v.p;
        CCL(ncclBroadcast(column?column:raw,raw,g.nr,Grid::device_type<T>(),owner,g.ccl_row,nullptr));
        int r0=std::upper_bound(g.rows.begin(),g.rows.end(),k)-g.rows.begin();
        int c0=std::upper_bound(g.cols.begin(),g.cols.end(),k)-g.cols.begin();
        int diagonal=(k/g.b)%g.pr==g.r?g.li(k,g.pr):-1;
        int alpha=((k+1)/g.b)%g.pr==g.r?g.li(k+1,g.pr):-1;
        local_statistics<<<1,256>>>(raw,g.nr,r0+(alpha>=0),diagonal,alpha,scalars.p,dd.p,ee.p,tt.p,k,g.pr==1);
        if (g.pr>1) {
            CCL(ncclAllGather(scalars.p,statistics.p,4,ncclFloat64,g.ccl_col,nullptr));
            reflector_scalar<<<1,1>>>(statistics.p,g.pr,scalars.p,dd.p,ee.p,tt.p,k);
            gather(v);
        }
        form_vectors<<<(std::max(padded,g.nc)+255)/256,256>>>(v.p,vc.p,w.p,column,raw,packed.p,
            rows.p,cols.p,g.nr,g.nc,padded,g.b,g.pr,k,scalars.p,tt.p);
        if (r0<g.nr && c0<g.nc)
            gemv(h,g.nr-r0,g.nc-c0,g.nr,a.p+r0+size_t(c0)*g.nr,vc.p+c0,w.p+r0);
        g.device_sum(w,g.row);
        vector_dot<<<1,256>>>(v.p,w.p,g.nr,r0,k,scalars.p,tt.p);
        if (g.pr>1) {
            CCL(ncclAllReduce(scalars.p+4,scalars.p+4,1,ncclFloat64,ncclSum,g.ccl_col,nullptr));
            gather(w);
        }
        T* corrected=g.pr==1?packed.p:w.p;
        correct_vectors<<<(std::max(padded,g.nc)+255)/256,256>>>(corrected,wc.p,w.p,v.p,vc.p,
            g.pr==1?w.p:packed.p,cols.p,g.nc,padded,g.b,g.pr,scalars.p,tt.p,k);
        int mr=g.nr-r0, mc=g.nc-c0;
        if (mr && mc)
            local_rank2<<<dim3((mr+127)/128,(mc+1)/2),dim3(128,2)>>>(a.p,v.p,corrected,vc.p,wc.p,g.nr,g.nc,r0,c0);
        HIP(hipGetLastError());
    }
    d=dd.get(); e=ee.get(); tau=tt.get();
    T native_last=0;
    if (g.rows.back()==g.n-1 && g.cols.back()==g.n-1)
        HIP(hipMemcpy(&native_last,a.p+g.nr-1+size_t(g.nc-1)*g.nr,sizeof(T),hipMemcpyDeviceToHost));
    double last=double(native_last);
    g.sum(&last,1,MPI_COMM_WORLD); d.back()=last;
}

template<class T> __global__ void matrix_max(const T* a, size_t n, double* out) {
    __shared__ double partial[256];
    int t=threadIdx.x; double value=0;
    for (size_t i=blockIdx.x*size_t(blockDim.x)+t;i<n;i+=size_t(gridDim.x)*blockDim.x)
        value=fmax(value,fabs(double(a[i])));
    partial[t]=value; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (t<stride) partial[t]=fmax(partial[t],partial[t+stride]);
        __syncthreads();
    }
    if (!t) out[blockIdx.x]=partial[0];
}

template<class T> __global__ void matrix_scale(T* a, size_t n, int exponent) {
    size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (i<n) a[i]=T(scalbn(double(a[i]),exponent));
}

void elpa_check(int error) {
    if (error!=ELPA_OK) throw std::runtime_error(elpa_strerr(error));
}

template<class T> void tridiagonal(Grid& g, Grid& target, std::vector<double>& d,
    std::vector<double>& e, Device<T>& q) {
    elpa_check(elpa_init(ELPA_API_VERSION));
    int error;
    auto handle=elpa_allocate(&error); elpa_check(error);
    auto set=[&](const char* key, int value) {
        elpa_set_integer(handle,key,value,&error); elpa_check(error);
    };
    set("na",g.n); set("nev",g.n); set("local_nrows",target.nr); set("local_ncols",target.nc);
    set("nblk",g.b); set("mpi_comm_parent",int(MPI_Comm_c2f(MPI_COMM_WORLD)));
    set("process_row",target.r); set("process_col",target.c);
    elpa_check(elpa_setup(handle));
    set("solver",ELPA_SOLVER_1STAGE); set("amd-gpu",1); set("use_gpu_id",0);
    elpa_check(elpa_setup_gpu(handle));
    // ELPA's public tridiagonal component accepts host-distributed arrays.
    std::vector<T> z(size_t(target.nr)*target.nc), output(q.n);
    if constexpr(std::is_same_v<T,double>)
        elpa_solve_tridiagonal_d(handle,d.data(),e.data(),z.data(),&error);
    else {
        std::vector<float> df(d.begin(),d.end()), ef(e.begin(),e.end());
        elpa_solve_tridiagonal_f(handle,df.data(),ef.data(),z.data(),&error);
        d.assign(df.begin(),df.end());
    }
    elpa_check(error);
    elpa_deallocate(handle,&error); elpa_check(error);
    elpa_uninit(&error); elpa_check(error);
    int one=1;
    if constexpr(std::is_same_v<T,double>)
        pdgemr2d_(&g.n,&g.n,z.data(),&one,&one,target.desc,output.data(),&one,&one,g.desc,&g.ctxt);
    else psgemr2d_(&g.n,&g.n,z.data(),&one,&one,target.desc,output.data(),&one,&one,g.desc,&g.ctxt);
    q.put(output);
}

template<class T> void eigh(Grid& g, Grid& target, rocblas_handle h,
    Device<T>& a, Device<T>& q, std::vector<double>& values) {
    int groups=std::min<size_t>(1024,(a.n+255)/256), exponent=0;
    Device<double> maxima(groups);
    matrix_max<<<groups,256>>>(a.p,a.n,maxima.p);
    auto local=maxima.get();
    double maximum=g.maximum(*std::max_element(local.begin(),local.end()));
    std::frexp(maximum,&exponent);
    // Exact power-of-two scaling keeps FP32 tridiagonal work in a useful range.
    if (exponent) matrix_scale<<<(a.n+255)/256,256>>>(a.p,a.n,-exponent);
    std::vector<double> d(g.n), e(g.n), tau(g.n-1);
    reduce(g,h,a,d,e,tau);
    tridiagonal(g,target,d,e,q);
    values=d;
    for (double& value:values) value=std::scalbn(value,exponent);
    backtransform(g,h,a,q,tau);
}

double entry(int i, int j, int n, const std::string& matrix) {
    if (matrix=="zero") return 0;
    if (matrix=="diagonal") return i==j?double((i*17)%n)/n:0;
    if (matrix=="clustered") return i==j?1.+1e-9*(i%3):0;
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
    return val+(i==j?3.+double(i)/n:0);
}

// Distributed FP64 SUMMA for validation only, outside solver timings.
std::vector<double> product(Grid& g, rocblas_handle h,
    const std::vector<double>& a, const std::vector<double>& b) {
    Device<double> da(size_t(g.nr)*g.b), db(size_t(g.b)*g.nc), out(a.size());
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
        gemm(h,false,g.nr,g.nc,width,1.,da.p,g.nr,db.p,width,1.,out.p,g.nr);
    }
    return out.get();
}

template<class T> void run(Grid& g, Grid& target, const std::string& name, int repeat) {
    if (repeat<1) throw std::runtime_error("repeat must be positive");
    rocblas_handle h; BLAS(rocblas_create_handle(&h));
    std::vector<double> original(size_t(g.nr)*g.nc);
    for (int j=0;j<g.nc;++j) for (int i=0;i<g.nr;++i)
        original[i+size_t(j)*g.nr]=entry(g.rows[i],g.cols[j],g.n,name);
    std::vector<T> input(original.begin(),original.end());
    Device<T> a(input.size()), q(input.size());
    std::vector<double> values(g.n);
    for (int trial=0;trial<repeat;++trial) {
        a.put(input);
        double begin=g.tick();
        eigh(g,target,h,a,q,values);
        double seconds=g.maximum(g.tick()-begin);
        if (!g.rank) std::printf("TIMING n=%d grid=%dx%d precision=%s trial=%d seconds=%.9g\n",
            g.n,g.pr,g.pc,std::is_same_v<T,double>?"fp64":"fp32",trial,seconds);
    }
    auto native_q=q.get(); std::vector<double> qd(native_q.begin(),native_q.end());
    auto aq=product(g,h,original,qd);
    std::vector<double> trans(qd.size());
    int one=1; double alpha=1, beta=0;
    pdtran_(&g.n,&g.n,&alpha,qd.data(),&one,&one,g.desc,&beta,trans.data(),&one,&one,g.desc);
    auto gram=product(g,h,trans,qd);
    double sums[3]={};
    for (int j=0;j<g.nc;++j) for (int i=0;i<g.nr;++i) {
        size_t z=i+size_t(j)*g.nr;
        double diff=aq[z]-qd[z]*values[g.cols[j]];
        double orth=gram[z]-(g.rows[i]==g.cols[j]?1.:0.);
        sums[0]+=diff*diff; sums[1]+=original[z]*original[z]; sums[2]+=orth*orth;
    }
    g.sum(sums,3,MPI_COMM_WORLD);
    double residual=std::sqrt(sums[0]/(sums[1]!=0?sums[1]:1));
    double orth=std::sqrt(sums[2]);
    double tolerance=std::is_same_v<T,double>?1e-10:1e-4;
    if (!std::isfinite(residual) || residual>tolerance || !std::isfinite(orth) || orth>tolerance*std::sqrt(g.n))
        throw std::runtime_error("residual or orthogonality check failed");
    if (!g.rank && g.n<=512) {
        std::vector<double> dense(size_t(g.n)*g.n), ref(g.n), work(3*g.n);
        for (int j=0;j<g.n;++j) for (int i=0;i<g.n;++i)
            dense[i+size_t(j)*g.n]=entry(i,j,g.n,name);
        int info, lwork=work.size();
        dsyev_("N","L",&g.n,dense.data(),&g.n,ref.data(),work.data(),&lwork,&info);
        if (info) throw std::runtime_error("LAPACK reference failed");
        double diff=0, scale=0;
        for (int i=0;i<g.n;++i) {
            diff=std::max(diff,std::abs(values[i]-ref[i])); scale=std::max(scale,std::abs(ref[i]));
        }
        double error=scale!=0?diff/scale:diff;
        if (!std::isfinite(error) || error>tolerance) throw std::runtime_error("eigenvalue check failed");
        std::printf("REFERENCE eigenvalue_error=%.9g\n",error);
    }
    if (!g.rank) std::printf("PASS n=%d grid=%dx%d block=%d precision=%s matrix=%s residual=%.9g orthogonality=%.9g\n",
        g.n,g.pr,g.pc,g.b,std::is_same_v<T,double>?"fp64":"fp32",name.c_str(),residual,orth);
    BLAS(rocblas_destroy_handle(h));
}

int main(int argc, char** argv) {
    // This Frontier stack requires HIP initialization before MPI.
    HIP(hipSetDevice(0)); HIP(hipFree(nullptr));
    MPI(MPI_Init(&argc,&argv));
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        int n=512, block=64, pr=1, pc=size, repeat=1;
        std::string precision="fp64", matrix="random";
        for (int i=1;i<argc;i+=2) {
            if (i+1==argc) throw std::runtime_error("missing option value");
            std::string key=argv[i], value=argv[i+1];
            if (key=="--n") n=std::stoi(value);
            else if (key=="--block") block=std::stoi(value);
            else if (key=="--pr") pr=std::stoi(value);
            else if (key=="--pc") pc=std::stoi(value);
            else if (key=="--repeat") repeat=std::stoi(value);
            else if (key=="--precision") precision=value;
            else if (key=="--matrix") matrix=value;
            else throw std::runtime_error("unknown option: "+key);
        }
        {
            Grid g(n,block,pr,pc), target(n,block,size,1);
            if (precision=="fp64") run<double>(g,target,matrix,repeat);
            else if (precision=="fp32") run<float>(g,target,matrix,repeat);
            else throw std::runtime_error("precision must be fp64 or fp32");
        }
        MPI(MPI_Finalize()); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what());
        MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
