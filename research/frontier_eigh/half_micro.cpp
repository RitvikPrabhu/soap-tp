// Single-GCD bulk_gemm diagnostic. No solver/MPI dependency; column-major throughout.
// Pooled phases exclude allocation, initialization, and C reset. The separate
// alloc+pack+gemm+free phases include per-call synchronous hipMalloc/hipFree.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>
#include "half_fused.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <rocwmma/rocwmma-version.hpp>

#define HIP(x) do { auto status_=(x); if(status_!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_)); } while(0)
#define BLAS(x) do { auto status_=(x); if(status_!=rocblas_status_success) throw std::runtime_error(std::string(#x)+": "+std::to_string(status_)); } while(0)
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
    ~Buffer() { (void)hipFree(p); }
    Buffer(const Buffer&)=delete;
};
struct Shape { const char* name; int m,n,k; bool ta,tb; float alpha,beta; int pad=0; };

// Deterministic, asymmetric inputs, bounded even in the dynamic-range test.
__host__ __device__ float value(size_t z, unsigned seed, int distribution, int k) {
    uint32_t u=uint32_t(z)*747796405u+seed*2891336453u;
    u=((u>>((u>>28)+4))^u)*277803737u; u=(u>>22)^u;
    float x=(float(u&65535)/32768.f-1.f);
    if(distribution==1) return ldexpf(x,int((u>>16)%25)-20); // ~2^-20 .. 2^4
    return x/sqrtf(float(k)); // eigenvector/Householder-like scale, not solver data
}
__global__ void initialize(float* x,size_t count,unsigned seed,int distribution,int k) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if(z<count) x[z]=value(z,seed,distribution,k);
}
// Exact current split-half layout and subnormal routing from eigh.cpp::to_half.
__global__ void pack(__half* out,const float* in,int rows,int cols,int ld,
                     int parts,bool stack_rows,bool swap) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if(z>=size_t(rows)*cols) return;
    float x=in[z%rows+(z/rows)*ld];
    __half hi=__float2half(parts==2 && fabsf(x)<0x1p-14f ? 0.f : x);
    size_t h=parts==2 && stack_rows ? z%rows+2*(z/rows)*rows : z;
    size_t l=h+(stack_rows?rows:size_t(rows)*cols);
    if(parts==2 && swap) { size_t t=h; h=l; l=t; }
    out[h]=hi;
    if(parts==2) out[l]=__float2half((x-__half2float(hi))*4096.f);
}

// Bounded experiment only: generated |x|<=16 keeps 64*hi and 4096*residual
// finite in half. NOT generic-range packing (64*hi can overflow for |hi|>1023.5).
// [64 Ah, Al, Ah] * [64 Bh; Bh; Bl] / 4096
//   = Ah*Bh + (Al*Bh + Ah*Bl)/4096. Low*low is still omitted.
__host__ __device__ __half concat64_value(float x,bool operand_a,int piece) {
    __half hi=__float2half(fabsf(x)<0x1p-14f?0.f:x);
    if(piece==0) return __float2half(64.f*__half2float(hi));
    if((operand_a && piece==1) || (!operand_a && piece==2))
        return __float2half((x-__half2float(hi))*4096.f);
    return hi;
}
__host__ __device__ size_t concat64_index(size_t z,int rows,int cols,bool stack,int piece) {
    return stack?z%rows+size_t(3)*(z/rows)*rows+size_t(piece)*rows:
                 z+size_t(piece)*rows*cols;
}
__global__ void pack_concat64(__half* out,const float* in,int rows,int cols,int ld,
                             bool stack,bool operand_a) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if(z>=size_t(rows)*cols) return;
    float x=in[z%rows+(z/rows)*ld];
    for(int piece=0;piece<3;++piece)
        out[concat64_index(z,rows,cols,stack,piece)]=concat64_value(x,operand_a,piece);
}

// Host-only check of actual packing helpers, all four layouts, tails/padding,
// and the balanced-scaling identity. No HIP runtime/device call is made.
void cpu_concat64_check() {
    for(bool ta:{false,true}) for(bool tb:{false,true}) for(int distribution:{0,1}) {
        const int m=17,n=23,k=35,ar=ta?k:m,ac=ta?m:k,br=tb?n:k,bc=tb?k:n;
        const int lda=ar+3,ldb=br+5,la=ta?3*ar:ar,lb=tb?br:3*br;
        std::vector<float> a(size_t(lda)*ac),b(size_t(ldb)*bc);
        std::vector<__half> ah(3*size_t(ar)*ac),bh(3*size_t(br)*bc);
        for(size_t z=0;z<a.size();++z) a[z]=value(z,11,distribution,k);
        for(size_t z=0;z<b.size();++z) b[z]=value(z,29,distribution,k);
        for(size_t z=0;z<size_t(ar)*ac;++z) for(int p=0;p<3;++p)
            ah[concat64_index(z,ar,ac,ta,p)]=concat64_value(a[z%ar+(z/ar)*lda],true,p);
        for(size_t z=0;z<size_t(br)*bc;++z) for(int p=0;p<3;++p)
            bh[concat64_index(z,br,bc,!tb,p)]=concat64_value(b[z%br+(z/br)*ldb],false,p);
        for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
            double packed=0,expected=0,bound=0;
            for(int l=0;l<3*k;++l) {
                double av=__half2float(ah[ta?l+size_t(i)*la:i+size_t(l)*la]);
                double bv=__half2float(bh[tb?j+size_t(l)*lb:l+size_t(j)*lb]);
                packed+=av*bv/4096.;
            }
            for(int l=0;l<k;++l) {
                float av=a[ta?l+size_t(i)*lda:i+size_t(l)*lda];
                float bv=b[tb?j+size_t(l)*ldb:l+size_t(j)*ldb];
                double ha=__half2float(__float2half(fabsf(av)<0x1p-14f?0.f:av));
                double hb=__half2float(__float2half(fabsf(bv)<0x1p-14f?0.f:bv));
                double lo_a=__half2float(__float2half((av-float(ha))*4096.f));
                double lo_b=__half2float(__float2half((bv-float(hb))*4096.f));
                double term=ha*hb+(lo_a*hb+ha*lo_b)/4096.;
                expected+=term; bound+=std::abs(ha*hb)+(std::abs(lo_a*hb)+std::abs(ha*lo_b))/4096.;
            }
            if(!std::isfinite(packed) || std::abs(packed-expected)>1e-12*bound+1e-30)
                throw std::runtime_error("CPU concat64 layout/scaling identity failed");
        }
        printf("CPU_CONCAT64_PASS transA=%c transB=%c distribution=%d\n",ta?'T':'N',tb?'T':'N',distribution);
    }
}
struct Events {
    hipEvent_t a,b;
    Events() { HIP(hipEventCreate(&a)); HIP(hipEventCreate(&b)); }
    ~Events() { (void)hipEventDestroy(a); (void)hipEventDestroy(b); }
    template<class F> float time(F f) {
        HIP(hipEventRecord(a,nullptr)); f(); HIP(hipEventRecord(b,nullptr));
        HIP(hipEventSynchronize(b)); float ms; HIP(hipEventElapsedTime(&ms,a,b)); return ms;
    }
};
struct Case {
    Shape s; int ar,ac,br,bc,lda,ldb,ldc;
    size_t an,bn,cn;
    Buffer<float> a,b,c,c0;
    Buffer<__half> ah,bh;
    rocblas_handle h;
    Case(Shape shape,rocblas_handle handle,int distribution,bool concat64=false): s(shape),
        ar(s.ta?s.k:s.m),ac(s.ta?s.m:s.k),br(s.tb?s.n:s.k),bc(s.tb?s.k:s.n),
        lda(ar+s.pad),ldb(br+s.pad),ldc(s.m+s.pad),
        an(size_t(lda)*ac),bn(size_t(ldb)*bc),cn(size_t(ldc)*s.n),
        a(an),b(bn),c(cn),c0(cn),ah((concat64?3:2)*size_t(ar)*ac),bh((concat64?3:2)*size_t(br)*bc),h(handle) {
        initialize<<<(an+255)/256,256>>>(a.p,an,11,distribution,s.k);
        initialize<<<(bn+255)/256,256>>>(b.p,bn,29,distribution,s.k);
        initialize<<<(cn+255)/256,256>>>(c0.p,cn,47,distribution,s.k);
        HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    }
    void reset() { HIP(hipMemcpyAsync(c.p,c0.p,cn*sizeof(float),hipMemcpyDeviceToDevice)); }
    void packing(int parts) {
        if(parts==3 || parts==4) return; // Fused modes pack inside their kernels.
        size_t na=size_t(ar)*ac,nb=size_t(br)*bc;
        if(parts==5) {
            pack_concat64<<<(na+255)/256,256>>>(ah.p,a.p,ar,ac,lda,s.ta,true);
            pack_concat64<<<(nb+255)/256,256>>>(bh.p,b.p,br,bc,ldb,!s.tb,false);
            HIP(hipGetLastError()); return;
        }
        pack<<<(na+255)/256,256>>>(ah.p,a.p,ar,ac,lda,parts,s.ta,false);
        pack<<<(nb+255)/256,256>>>(bh.p,b.p,br,bc,ldb,parts,!s.tb,true);
        HIP(hipGetLastError());
    }
    void product(int parts,unsigned flags) {
        if(parts==4) {
            half_fused::launch_tile(s.ta,s.tb,s.m,s.n,s.k,s.alpha,a.p,lda,b.p,ldb,s.beta,c.p,ldc);
            HIP(hipGetLastError()); return;
        }
        if(parts==3) {
            half_fused::launch(s.ta,s.tb,s.m,s.n,s.k,s.alpha,a.p,lda,b.p,ldb,s.beta,c.p,ldc);
            HIP(hipGetLastError()); return;
        }
        auto oa=s.ta?rocblas_operation_transpose:rocblas_operation_none;
        auto ob=s.tb?rocblas_operation_transpose:rocblas_operation_none;
        if(parts==5) {
            float alpha=s.alpha/4096.f;
            BLAS(rocblas_gemm_ex(h,oa,ob,s.m,s.n,3*s.k,&alpha,
                ah.p,rocblas_datatype_f16_r,s.ta?3*ar:ar,
                bh.p,rocblas_datatype_f16_r,s.tb?br:3*br,
                &s.beta,c.p,rocblas_datatype_f32_r,ldc,c.p,rocblas_datatype_f32_r,ldc,
                rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,flags)); return;
        }
        if(!parts) {
            BLAS(rocblas_sgemm(h,oa,ob,s.m,s.n,s.k,&s.alpha,a.p,lda,b.p,ldb,
                               &s.beta,c.p,ldc)); return;
        }
        int la=parts==2 && s.ta?2*ar:ar,lb=parts==2 && !s.tb?2*br:br;
        size_t bhi=parts==2?(s.tb?size_t(br)*bc:br):0;
        for(int term=0;term<parts;++term) {
            float alpha=term?s.alpha/4096.f:s.alpha,beta=term?1.f:s.beta;
            BLAS(rocblas_gemm_ex(h,oa,ob,s.m,s.n,term?2*s.k:s.k,
                &alpha,ah.p,rocblas_datatype_f16_r,la,
                bh.p+(term?0:bhi),rocblas_datatype_f16_r,lb,
                &beta,c.p,rocblas_datatype_f32_r,ldc,c.p,rocblas_datatype_f32_r,ldc,
                rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,flags));
        }
    }
    void error(const char* mode,int distribution,int parts) {
        // Independent CPU FP64 dot products from the original FP32 operands.
        // Only called for tiny shapes; no target-size FP64 allocation/reference.
        if(parts>=3) parts=2; // Same three algebraic products, different accumulation order.
        std::vector<float> ha(an),hb(bn),hc(cn),initial(cn);
        HIP(hipMemcpy(ha.data(),a.p,an*sizeof(float),hipMemcpyDeviceToHost));
        HIP(hipMemcpy(hb.data(),b.p,bn*sizeof(float),hipMemcpyDeviceToHost));
        HIP(hipMemcpy(hc.data(),c.p,cn*sizeof(float),hipMemcpyDeviceToHost));
        HIP(hipMemcpy(initial.data(),c0.p,cn*sizeof(float),hipMemcpyDeviceToHost));
        for(int j=0;j<s.n;++j) for(int i=s.m;i<ldc;++i)
            if(hc[i+size_t(j)*ldc]!=initial[i+size_t(j)*ldc])
                throw std::runtime_error("C padding was overwritten");
        double err2=0,ref2=0,maxabs=0,scaled=0;
        for(int j=0;j<s.n;++j) for(int i=0;i<s.m;++i) {
            double dot=0,bound=0,represented=0,represented_bound=0,flush_uncertainty=0;
            for(int l=0;l<s.k;++l) {
                double av=ha[s.ta?l+size_t(i)*lda:i+size_t(l)*lda];
                double bv=hb[s.tb?j+size_t(l)*ldb:l+size_t(j)*ldb];
                dot+=av*bv; bound+=std::abs(av*bv);
                auto half=[](double x) {
                    float q=__half2float(__float2half(float(x)));
                    return std::abs(q)<0x1p-14f?0.:double(q); // MI200 input FTZ
                };
                double term=av*bv;
                if(parts) {
                    double hi_a=parts==2 && std::abs(av)<0x1p-14?0.:__half2float(__float2half(float(av)));
                    double hi_b=parts==2 && std::abs(bv)<0x1p-14?0.:__half2float(__float2half(float(bv)));
                    term=half(hi_a)*half(hi_b);
                    flush_uncertainty+=std::abs(hi_a*hi_b-term);
                    if(parts==2) {
                        double lo_a=__half2float(__float2half(float((av-hi_a)*4096.)));
                        double lo_b=__half2float(__float2half(float((bv-hi_b)*4096.)));
                        double cross1=half(hi_a)*half(lo_b),cross2=half(lo_a)*half(hi_b);
                        term+=(cross1+cross2)/4096.;
                        flush_uncertainty+=(std::abs(hi_a*lo_b-cross1)+std::abs(lo_a*hi_b-cross2))/4096.;
                    }
                }
                represented+=term; represented_bound+=std::abs(term);
            }
            double ref=double(s.alpha)*dot+double(s.beta)*initial[i+size_t(j)*ldc];
            double e=std::abs(double(hc[i+size_t(j)*ldc])-ref);
            if(!std::isfinite(e)) throw std::runtime_error("nonfinite numerical result");
            err2+=e*e; ref2+=ref*ref; maxabs=std::max(maxabs,e);
            bound=std::abs(s.alpha)*bound+std::abs(double(s.beta)*initial[i+size_t(j)*ldc]);
            scaled=std::max(scaled,e/std::max(bound,1e-300));
            double model=double(s.alpha)*represented+double(s.beta)*initial[i+size_t(j)*ldc];
            double model_bound=std::abs(s.alpha)*represented_bound+
                               std::abs(double(s.beta)*initial[i+size_t(j)*ldc]);
            // Validate layout/algebra against represented operands, not an accuracy
            // promise: when both highs vanish, omitted low*low can be all of AB.
            // Tiny rocBLAS fallback kernels can preserve FP16 subnormals while
            // MFMA kernels flush them: allow the analytically bounded difference.
            double tolerance=std::abs(s.alpha)*flush_uncertainty+
                             32.*s.k*double(0x1p-23f)*(model_bound+flush_uncertainty)+1e-30;
            if(std::abs(double(hc[i+size_t(j)*ldc])-model)>tolerance)
                throw std::runtime_error("tiny represented-product algebra mismatch");
        }
        printf("ERROR,%s,%s,dist=%d,rel_frob=%.9g,max_abs=%.9g,max_bound_scaled=%.9g\n",
               s.name,mode,distribution,sqrt(err2/std::max(ref2,1e-300)),maxabs,scaled);
    }
};
struct Mode { const char* name; int parts; unsigned flags; };
void stats(const Shape& s,const Mode& mode,const char* phase,std::vector<float> x);

// Research-only beta API: NULL list queries count; a list of that capacity
// returns supported indices for the exact pointers/strides/scalars/types.
// Explicit indices require algo_solution_index, not algo_standard.
struct TuneGemm {
    Case& c; const char* name; int k,la,lb; float alpha,beta;
    const void* a; const void* b; rocblas_datatype type;
    // Only our exact -1 sentinel means heuristic selection. Returned Tensile
    // IDs are negative too: preserve their sign/value and use explicit selection.
    static constexpr rocblas_gemm_algo algorithm(int solution) {
        return solution==-1?rocblas_gemm_algo_standard:rocblas_gemm_algo_solution_index;
    }
    static constexpr int requested_index(int solution) { return solution==-1?0:solution; }
    rocblas_status invoke(int solution,rocblas_int* list=nullptr,rocblas_int* count=nullptr) const {
        auto oa=c.s.ta?rocblas_operation_transpose:rocblas_operation_none;
        auto ob=c.s.tb?rocblas_operation_transpose:rocblas_operation_none;
        if(count) return rocblas_gemm_ex_get_solutions(c.h,oa,ob,c.s.m,c.s.n,k,&alpha,
            a,type,la,b,type,lb,&beta,c.c.p,rocblas_datatype_f32_r,c.ldc,
            c.c.p,rocblas_datatype_f32_r,c.ldc,rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index,0,list,count);
        return rocblas_gemm_ex(c.h,oa,ob,c.s.m,c.s.n,k,&alpha,a,type,la,b,type,lb,
            &beta,c.c.p,rocblas_datatype_f32_r,c.ldc,c.c.p,rocblas_datatype_f32_r,c.ldc,
            rocblas_datatype_f32_r,algorithm(solution),requested_index(solution),0);
    }
    std::vector<rocblas_int> solutions() const {
        rocblas_int count=0;
        BLAS(invoke(-1,nullptr,&count)); // Explicit tuning requires working beta API.
        if(count<0) throw std::runtime_error("solution query returned negative count");
        std::vector<rocblas_int> all(count);
        if(count) { rocblas_int capacity=count; BLAS(invoke(-1,all.data(),&capacity)); }
        return all;
    }
};
static_assert(TuneGemm::algorithm(-1)==rocblas_gemm_algo_standard && TuneGemm::requested_index(-1)==0);
static_assert(TuneGemm::algorithm(-606618098)==rocblas_gemm_algo_solution_index &&
              TuneGemm::requested_index(-606618098)==-606618098);
static_assert(TuneGemm::algorithm(123)==rocblas_gemm_algo_solution_index && TuneGemm::requested_index(123)==123);

// Gather only 12 complete dot products, including corners/interior, for an
// independent CPU FP64 reference at the exact target shape. Never full FP64 C.
__global__ void gather_tune_dots(float* out,int m,int n,int k,bool ta,bool tb,
    const void* a,int lda,const void* b,int ldb,bool ishalf,
    const float* initial,const float* result,int ldc) {
    size_t z=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if(z>=size_t(12)*k) return;
    int q=z/k,l=z%k;
    int i=q==0?0:q==1?m-1:int((size_t(q)*104729)%m);
    int j=q==0?0:q==1?n-1:int((size_t(q)*13007)%n);
    size_t ai=ta?l+size_t(i)*lda:i+size_t(l)*lda;
    size_t bi=tb?j+size_t(l)*ldb:l+size_t(j)*ldb;
    out[z]=ishalf?__half2float(static_cast<const __half*>(a)[ai]):static_cast<const float*>(a)[ai];
    out[12*size_t(k)+z]=ishalf?__half2float(static_cast<const __half*>(b)[bi]):static_cast<const float*>(b)[bi];
    if(l==0) { out[24*size_t(k)+q]=initial[i+size_t(j)*ldc]; out[24*size_t(k)+12+q]=result[i+size_t(j)*ldc]; }
}
void validate_tuned(const TuneGemm& g,bool original_reference=false) {
    Buffer<float> sample(24*size_t(g.k)+24);
    gather_tune_dots<<<(12*size_t(g.k)+255)/256,256>>>(sample.p,g.c.s.m,g.c.s.n,g.k,
        g.c.s.ta,g.c.s.tb,g.a,g.la,g.b,g.lb,g.type==rocblas_datatype_f16_r,g.c.c0.p,g.c.c.p,g.c.ldc);
    HIP(hipGetLastError());
    std::vector<float> x(24*size_t(g.k)+24);
    HIP(hipMemcpy(x.data(),sample.p,x.size()*sizeof(float),hipMemcpyDeviceToHost));
    double maxscaled=0,err2=0,ref2=0;
    for(int q=0;q<12;++q) {
        double dot=0,bound=0;
        for(int l=0;l<g.k;++l) {
            double term=double(x[q*size_t(g.k)+l])*x[12*size_t(g.k)+q*size_t(g.k)+l];
            dot+=term; bound+=std::abs(term);
        }
        double ref=double(g.alpha)*dot+double(g.beta)*x[24*size_t(g.k)+q];
        bound=std::abs(g.alpha)*bound+std::abs(double(g.beta)*x[24*size_t(g.k)+q]);
        double err=std::abs(double(x[24*size_t(g.k)+12+q])-ref);
        if(!std::isfinite(err)) throw std::runtime_error("nonfinite tuned result");
        maxscaled=std::max(maxscaled,err/std::max(bound,1e-300)); err2+=err*err; ref2+=ref*ref;
        // Packed FP16 subnormals may be flushed by MFMA: bound that input
        // discrepancy independently, rather than treating it as layout failure.
        double flush=0;
        if(g.type==rocblas_datatype_f16_r) for(int l=0;l<g.k;++l) {
            double av=x[q*size_t(g.k)+l],bv=x[12*size_t(g.k)+q*size_t(g.k)+l];
            if(std::abs(av)<0x1p-14 || std::abs(bv)<0x1p-14) flush+=std::abs(av*bv);
        }
        if(!original_reference && err>std::abs(g.alpha)*flush+32.*g.k*0x1p-23*bound+1e-30)
            throw std::runtime_error("chosen solution disagrees with sampled independent FP64 products");
    }
    printf("TUNE_CHECK,%s,%s,reference=%s,samples=12,rel_frob=%.9g,max_bound_scaled=%.9g\n",
        g.c.s.name,g.name,original_reference?"original_FP64":"packed_FP64",sqrt(err2/std::max(ref2,1e-300)),maxscaled);
}

int tune_one(const TuneGemm& g,int cap,std::chrono::steady_clock::time_point deadline) {
    printf("TUNE_PROBLEM,%s,%s,m=%d,n=%d,k=%d,lda=%d,ldb=%d,ldc=%d,alpha=%.9g,beta=%.9g,input=%s,flags=0\n",
        g.c.s.name,g.name,g.c.s.m,g.c.s.n,g.k,g.la,g.lb,g.c.ldc,g.alpha,g.beta,
        g.type==rocblas_datatype_f16_r?"f16":"f32");
    auto all=g.solutions();
    int count=all.size();
    Events e;
    auto measure=[&](int id) {
        g.c.reset(); BLAS(g.invoke(id)); HIP(hipDeviceSynchronize()); // warm this exact solution
        std::vector<float> t;
        for(int r=0;r<3;++r) {
            g.c.reset(); HIP(hipDeviceSynchronize());
            t.push_back(e.time([&]{BLAS(g.invoke(id));}));
        }
        std::sort(t.begin(),t.end()); return t[1];
    };
    float best=measure(-1); int winner=-1,tried=0,failed=0;
    printf("TUNE_CANDIDATE,%s,%s,solution=standard,median_ms=%.6f\n",g.c.s.name,g.name,best);
    int selected=std::min(cap,int(all.size()));
    for(int p=0;p<selected;++p) {
        if(std::chrono::steady_clock::now()>=deadline) break;
        // Evenly sample the complete returned list, no guessed preferred IDs.
        int id=all[selected==1?0:size_t(p)*(all.size()-1)/(selected-1)];
        g.c.reset(); auto ok=g.invoke(id);
        if(ok!=rocblas_status_success) {
            printf("TUNE_REJECT,%s,%s,solution=%d,status=%d\n",g.c.s.name,g.name,id,int(ok)); ++failed; continue;
        }
        HIP(hipDeviceSynchronize());
        float ms=measure(id); ++tried;
        printf("TUNE_CANDIDATE,%s,%s,solution=%d,median_ms=%.6f\n",g.c.s.name,g.name,id,ms);
        if(ms<best) { best=ms; winner=id; }
    }
    printf("TUNE_BEST,%s,%s,supported=%d,cap=%d,timed=%d,rejected=%d,untried=%d,solution=%d,median_ms=%.6f\n",
        g.c.s.name,g.name,count,cap,tried,failed,count-tried-failed,winner,best);
    g.c.reset(); BLAS(g.invoke(winner)); HIP(hipDeviceSynchronize()); validate_tuned(g);
    return winner;
}

// No event or wall timing. Exactly three labelled GEMM requests, each followed
// by completion and an independent sampled FP64 product check. Main can profile
// actual dispatches; labels alone do not establish distinct selected kernels.
void selection_check(rocblas_handle h) {
    Case c({"back1_8192",256,1024,8192,true,false,1,0},h,0,true);
    c.packing(5); HIP(hipDeviceSynchronize());
    TuneGemm g{c,"concat64",3*c.s.k,3*c.ar,3*c.br,c.s.alpha/4096.f,c.s.beta,
               c.ah.p,c.bh.p,rocblas_datatype_f16_r};
    auto ids=g.solutions();
    if(ids.empty()) throw std::runtime_error("selection-check requires at least one returned solution ID");
    printf("SELECTION_CHECK,shape=back1_8192,mode=concat64,m=256,n=1024,k=24576,supported=%zu,first=%d,last=%d,no_timing=1\n",
           ids.size(),int(ids.front()),int(ids.back()));
    const char* labels[]={"standard","first","last"};
    int requests[]={-1,ids.front(),ids.back()};
    for(int q=0;q<3;++q) {
        int id=requests[q];
        c.reset(); HIP(hipDeviceSynchronize());
        printf("SELECTION_REQUEST,label=%s,requested_algo=%d,requested_index=%d,raw_id=%d\n",
               labels[q],int(TuneGemm::algorithm(id)),TuneGemm::requested_index(id),id);
        fflush(stdout);
        BLAS(g.invoke(id)); HIP(hipDeviceSynchronize());
        validate_tuned(g);
        printf("SELECTION_COMPLETE,label=%s,requested_index=%d,FP64_sample_check=passed\n",
               labels[q],TuneGemm::requested_index(id));
        fflush(stdout);
    }
}

void autotune_case(Case& c,int cap,std::chrono::steady_clock::time_point deadline) {
    TuneGemm native{c,"native_ex",c.s.k,c.lda,c.ldb,c.s.alpha,c.s.beta,c.a.p,c.b.p,rocblas_datatype_f32_r};
    int ns=tune_one(native,cap,deadline);
    c.packing(5); HIP(hipDeviceSynchronize());
    TuneGemm concat{c,"concat64",3*c.s.k,c.s.ta?3*c.ar:c.ar,c.s.tb?c.br:3*c.br,
        c.s.alpha/4096.f,c.s.beta,c.ah.p,c.bh.p,rocblas_datatype_f16_r};
    int cs=tune_one(concat,cap,deadline);
    c.packing(2); HIP(hipDeviceSynchronize());
    int la=c.s.ta?2*c.ar:c.ar,lb=c.s.tb?c.br:2*c.br;
    size_t bhi=c.s.tb?size_t(c.br)*c.bc:c.br;
    TuneGemm hh{c,"split_HH",c.s.k,la,lb,c.s.alpha,c.s.beta,c.ah.p,c.bh.p+bhi,rocblas_datatype_f16_r};
    TuneGemm cross{c,"split_cross",2*c.s.k,la,lb,c.s.alpha/4096.f,1.f,c.ah.p,c.bh.p,rocblas_datatype_f16_r};
    int hs=tune_one(hh,cap,deadline),xs=tune_one(cross,cap,deadline);
    Events e;
    for(int parts:{-1,0,5,2}) {
        std::vector<float> prepacked,total,wall,packtime,resetwall,inclusivewall;
        auto product=[&] { if(parts==-1) { c.product(0,0); }
            else if(parts==0) { BLAS(native.invoke(ns)); }
            else if(parts==5) { BLAS(concat.invoke(cs)); }
            else { BLAS(hh.invoke(hs)); BLAS(cross.invoke(xs)); } };
        if(parts>0) c.packing(parts);
        for(int r=0;r<5;++r) {
            auto resetstart=std::chrono::steady_clock::now(); c.reset(); HIP(hipDeviceSynchronize());
            resetwall.push_back(std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-resetstart).count());
            prepacked.push_back(e.time(product));
            c.reset(); HIP(hipDeviceSynchronize());
            if(parts>0) packtime.push_back(e.time([&]{c.packing(parts);}));
            c.reset(); HIP(hipDeviceSynchronize());
            total.push_back(e.time([&]{if(parts>0) c.packing(parts); product();}));
            // Wall excludes C reset, but includes launches and completion wait.
            c.reset(); HIP(hipDeviceSynchronize()); auto start=std::chrono::steady_clock::now();
            if(parts>0) c.packing(parts);
            product(); HIP(hipDeviceSynchronize());
            wall.push_back(std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count());
            start=std::chrono::steady_clock::now(); c.reset();
            if(parts>0) c.packing(parts);
            product(); HIP(hipDeviceSynchronize());
            inclusivewall.push_back(std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        Mode mode{parts==-1?"native_sgemm":parts==0?"tuned_native_ex":parts==5?"tuned_concat64":"tuned_split",parts,0};
        stats(c.s,mode,"prepacked_gemm",prepacked); stats(c.s,mode,"pack+gemm",total);
        stats(c.s,mode,"pack+gemm_wall",wall); stats(c.s,mode,"C_reset_wall_separate",resetwall);
        stats(c.s,mode,"C_reset+pack+gemm_wall",inclusivewall);
        if(parts>0) stats(c.s,mode,"pack",packtime);
        TuneGemm original=native; original.name=mode.name; validate_tuned(original,true);
    }
}
void stats(const Shape& s,const Mode& mode,const char* phase,std::vector<float> x) {
    std::sort(x.begin(),x.end()); double median=x[x.size()/2];
    // effective TF/s uses original 2mnk, not 3x split-half executed FLOPs.
    printf("TIME,%s,%s,%s,%d,%d,%d,%c,%c,%.6f,%.6f,%.6f,%.6f\n",s.name,mode.name,phase,
        s.m,s.n,s.k,s.ta?'T':'N',s.tb?'T':'N',x.front(),median,x.back(),
        (phase==std::string("pack") || phase==std::string("C_reset_wall_separate"))?0.:2.*s.m*s.n*s.k/(median*1e9));
}
int main(int argc,char** argv) try {
    bool check=false,concat64=false,cpu_concat=false,autotune=false,selection=false; int warm=3,repeat=10,tune_cap=8;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--check-only") check=true;
        else if(arg=="--concat64") concat64=true;
        else if(arg=="--cpu-concat-check") cpu_concat=true;
        else if(arg=="--selection-check") selection=true;
        else if(arg=="--autotune") { autotune=true; concat64=true; }
        else if(arg=="--autotune-max" && i+1<argc) {
            tune_cap=std::stoi(argv[++i]);
            if(tune_cap<1 || tune_cap>32) throw std::runtime_error("autotune-max must be 1..32");
        }
        else if((arg=="--warmup" || arg=="--repeat") && i+1<argc) {
            int v=std::stoi(argv[++i]); if(v<1 || v>100) throw std::runtime_error("count must be 1..100");
            (arg=="--warmup"?warm:repeat)=v;
        } else throw std::runtime_error("usage: half_micro [--check-only] [--concat64] [--cpu-concat-check] [--selection-check] [--autotune] [--autotune-max 1..32] [--warmup N] [--repeat N]");
    }
    if(cpu_concat) { cpu_concat64_check(); return 0; }
    if(!check && !std::getenv("SLURM_JOB_ID")) throw std::runtime_error("timing requires a Slurm batch allocation");
    int count,driver,runtime; HIP(hipGetDeviceCount(&count));
    if(count!=1) throw std::runtime_error("require exactly one visible GCD (srun --gpus-per-task=1)");
    HIP(hipSetDevice(0)); hipDeviceProp_t p; HIP(hipGetDeviceProperties(&p,0));
    HIP(hipDriverGetVersion(&driver)); HIP(hipRuntimeGetVersion(&runtime));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    printf("DEVICE name=%s arch=%s pci=%s memory=%zu CU=%d driver=%d runtime=%d HIP_VERSION=%d\n",
        p.name,p.gcnArchName,pci,p.totalGlobalMem,p.multiProcessorCount,driver,runtime,HIP_VERSION);
    if(std::string(p.gcnArchName).find("gfx90a")==std::string::npos)
        throw std::runtime_error("this experiment requires gfx90a/MI250X");
    if(!check && std::string(p.name).find("MI250X")==std::string::npos)
        throw std::runtime_error("target timings require MI250X; other gfx90a allowed only for tiny checks");
    rocblas_handle h; BLAS(rocblas_create_handle(&h));
    BLAS(rocblas_set_pointer_mode(h,rocblas_pointer_mode_host));
    BLAS(rocblas_set_stream(h,nullptr));
    char version[256]; BLAS(rocblas_get_version_string(version,sizeof(version)));
    Dl_info info{}; dladdr(dlsym(RTLD_DEFAULT,"rocblas_sgemm"),&info);
    printf("BACKEND rocblas=%s library=%s API=rocblas_sgemm/rocblas_gemm_ex algo=standard solution=0 compute=f32 stream=default\n",
           version,info.dli_fname?info.dli_fname:"unknown");
    if(selection) { selection_check(h); BLAS(rocblas_destroy_handle(h)); return 0; }
    printf("CONFIG warmup=%d repeat=%d allocation=pooled reset_C=outside_events high_half_is_NOT_FP32_accurate\n",warm,repeat);
    printf("FUSED_BACKEND rocWMMA=%d.%d.%d tile=16x16x16 wave=64 accumulators=HH+cross f32 C_writes=1 packing=inside_kernel\n",
           ROCWMMA_VERSION_MAJOR,ROCWMMA_VERSION_MINOR,ROCWMMA_VERSION_PATCH);
    printf("FUSED_TILE_BACKEND CTA=64x64x16 threads=256 wave_output=32x32 fragments=2x2 LDS_bytes=16384 C_writes=1\n");
    std::vector<Mode> modes={{"sgemm",0,0},{"half_hh",1,0},{"split_half",2,0},
                             {"fused_half",3,0},{"fused_half_tile",4,0}};
    if(concat64) {
        cpu_concat64_check();
        printf("CONCAT64_BOUND generated_abs_input<=16 NOT_generic_range K_factor=3 alpha_factor=1/4096 HH_scale=64x64\n");
        modes.push_back({"concat64",5,0});
    }
#ifdef HALF_MICRO_ALT
    modes.push_back({"half_hh_alt",1,rocblas_gemm_flags_fp16_alt_impl});
    modes.push_back({"split_half_alt",2,rocblas_gemm_flags_fp16_alt_impl});
#endif
    for(auto s:std::vector<Shape>{{"tiny_TN",13,19,37,true,false,1,0,3},
                                 {"tiny_NN",37,19,13,false,false,-1,1,5},
                                 {"tiny_NT",41,17,14,false,true,-1,1,7},
                                 {"tiny_TT",17,23,33,true,true,0.75f,-0.5f,3},
                                 {"tile_TN",67,73,35,true,false,1,0,3},
                                 {"tile_NN",73,67,19,false,false,-1,1,5},
                                 {"tile_NT",65,81,33,false,true,-1,1,7},
                                 {"tile_TT",71,69,17,true,true,0.75f,-0.5f,3}})
        for(int distribution=0;distribution<2;++distribution) {
            Case c(s,h,distribution,concat64);
            for(auto mode:modes) {
                c.reset(); if(mode.parts) c.packing(mode.parts);
                c.product(mode.parts,mode.flags); HIP(hipDeviceSynchronize()); c.error(mode.name,distribution,mode.parts);
            }
        }
    if(!check) {
        // Global exploration budget; warmup/validation and final five-repeat
        // comparisons continue afterward. Not a hard process timeout.
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(45);
        if(autotune) printf("TUNE_CONFIG cap=%d selection=evenly_spaced_supported_list exploration_budget_s=45 warmup=1 repeats=3 final_repeats=5 allocation=pooled native_tuning_API=gemm_ex_f32 standard_solution=-1\n",tune_cap);
        printf("TIME_HEADER,shape,mode,phase,m,n,k,transA,transB,min_ms,median_ms,max_ms,effective_TF_s\n");
        for(int n:{8192,16384}) for(auto s:std::vector<Shape>{
            {n==8192?"back1_8192":"back1_16384",256,n/8,n,true,false,1,0},
            {n==8192?"back2_8192":"back2_16384",n,n/8,256,false,false,-1,1},
            {n==8192?"panel_8192":"panel_16384",n,n/8,64,false,true,-1,1}}) {
            Case c(s,h,0,concat64); Events e;
            if(autotune) { autotune_case(c,tune_cap,deadline); fflush(stdout); continue; }
            for(auto mode:modes) {
                for(int w=0;w<warm;++w) {
                    c.reset(); if(mode.parts) c.packing(mode.parts); c.product(mode.parts,mode.flags);
                }
                HIP(hipDeviceSynchronize());
                std::vector<float> pack_ms,gemm_ms,total_ms;
                for(int r=0;r<repeat;++r) {
                    c.reset(); HIP(hipDeviceSynchronize());
                    if(mode.parts==1 || mode.parts==2 || mode.parts==5)
                        pack_ms.push_back(e.time([&]{c.packing(mode.parts);}));
                    gemm_ms.push_back(e.time([&]{c.product(mode.parts,mode.flags);}));
                    c.reset(); HIP(hipDeviceSynchronize());
                    total_ms.push_back(e.time([&]{if(mode.parts) c.packing(mode.parts); c.product(mode.parts,mode.flags);}));
                }
                if(mode.parts==1 || mode.parts==2 || mode.parts==5) stats(s,mode,"pack",pack_ms);
                stats(s,mode,(mode.parts==3 || mode.parts==4)?"fused_including_tile_pack":"gemm",gemm_ms);
                stats(s,mode,"pack+gemm",total_ms);
                if(mode.parts==2) {
                    auto allocated_call=[&] {
                        Buffer<__half> ah(2*size_t(c.ar)*c.ac),bh(2*size_t(c.br)*c.bc);
                        std::swap(ah.p,c.ah.p); std::swap(bh.p,c.bh.p);
                        c.packing(mode.parts); c.product(mode.parts,mode.flags);
                        HIP(hipDeviceSynchronize());
                        std::swap(ah.p,c.ah.p); std::swap(bh.p,c.bh.p);
                        // Local buffers free the fresh allocations on return.
                    };
                    for(int w=0;w<warm;++w) { c.reset(); allocated_call(); }
                    std::vector<float> inclusive_events,inclusive_wall;
                    for(int r=0;r<repeat;++r) {
                        c.reset(); HIP(hipDeviceSynchronize());
                        inclusive_events.push_back(e.time(allocated_call));
                        c.reset(); HIP(hipDeviceSynchronize());
                        auto start=std::chrono::steady_clock::now();
                        allocated_call();
                        inclusive_wall.push_back(std::chrono::duration<float,std::milli>(
                            std::chrono::steady_clock::now()-start).count());
                    }
                    stats(s,mode,"alloc+pack+gemm+free_event",inclusive_events);
                    stats(s,mode,"alloc+pack+gemm+free_wall",inclusive_wall);
                }
                fflush(stdout);
            }
        }
    }
    BLAS(rocblas_destroy_handle(h)); return 0;
} catch(const std::exception& e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
