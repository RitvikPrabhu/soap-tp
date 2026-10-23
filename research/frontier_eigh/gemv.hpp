#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <type_traits>

// Research-only no-transpose y=A*x (alpha=1, beta=0), FP32 or FP64 throughout.
// A points to the active submatrix: A[i+j*lda], 0<=i<m, 0<=j<n.
// Preconditions: m,n>=0, lda>=max(1,m); device buffers do not overlap.
// Caller owns T partials[m*ceil(n/256)] (none needed when m==0 or n==0).
// No allocations, host synchronization, atomics, streams, or rocBLAS fallback.
// Two launches trade extra partial traffic and latency for bounded column
// parallelism; actual bandwidth and break-even sizes require batch measurements.
namespace tiled_gemv {
template<class T>
__global__ __launch_bounds__(512) void partial_kernel(int m,int n,const T* a,int lda,
                                                    const T* x,T* partials) {
    __shared__ T cached_x[256];
    __shared__ T sums[4][128];
    const int row=blockIdx.x*128+threadIdx.x;
    const int group=threadIdx.y,begin=blockIdx.y*256;
    const int linear=group*128+threadIdx.x;
    if(linear<256) cached_x[linear]=begin+linear<n?x[begin+linear]:T(0);
    __syncthreads();
    T sum=0;
    // Each wave reads 64 adjacent rows from a column, irrespective of lda.
    // Four groups split the columns; reductions remain in the input precision.
    if(row<m) for(int col=group;col<256 && begin+col<n;col+=4)
        sum+=a[row+std::size_t(begin+col)*lda]*cached_x[col];
    sums[group][threadIdx.x]=sum;
    __syncthreads();
    if(group==0 && row<m)
        partials[row+std::size_t(blockIdx.y)*m]=
            (sums[0][threadIdx.x]+sums[1][threadIdx.x])+
            (sums[2][threadIdx.x]+sums[3][threadIdx.x]);
}

template<class T>
__global__ void finish_kernel(int m,int chunks,const T* partials,T* y) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=m) return;
    T sum=0;
    for(int chunk=0;chunk<chunks;++chunk) sum+=partials[row+std::size_t(chunk)*m];
    y[row]=sum; // n==0 overwrites y with zero without reading x/A/partials.
}

template<class T>
inline void launch(int m,int n,const T* a,int lda,const T* x,T* y,T* partials) {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,"FP32/FP64 only");
    if(m==0) return;
    const int chunks=n/256+(n%256!=0);
    if(chunks) partial_kernel<T><<<dim3(m/128+(m%128!=0),chunks),dim3(128,4)>>>(m,n,a,lda,x,partials);
    finish_kernel<T><<<m/256+(m%256!=0),256>>>(m,chunks,partials,y);
    // Caller checks hipGetLastError() and synchronizes at its own dependency boundary.
}

// Distinct single-launch experiment: one CTA owns 64 rows for the entire n.
// Eight wave64 column groups accumulate independently, then reduce in LDS.
// This removes global partial traffic and one launch, but also removes chunk
// parallelism: short m / long n may leave too few CTAs to fill the device.
template<class T>
__global__ __launch_bounds__(512) void single_kernel(int m,int n,const T* a,int lda,
                                                   const T* x,T* y) {
    __shared__ T sums[8][64];
    const int row=blockIdx.x*64+threadIdx.x,group=threadIdx.y;
    T sum=0;
    if(row<m) for(int col=group;col<n;col+=8)
        sum+=a[row+std::size_t(col)*lda]*x[col];
    sums[group][threadIdx.x]=sum;
    __syncthreads();
    if(group==0 && row<m) {
        T total=0;
        #pragma unroll
        for(int g=0;g<8;++g) total+=sums[g][threadIdx.x];
        y[row]=total; // n==0: all eight sums are zero; no A/x reads.
    }
}

template<class T>
inline void launch_single(int m,int n,const T* a,int lda,const T* x,T* y) {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,"FP32/FP64 only");
    if(m==0) return;
    single_kernel<T><<<m/64+(m%64!=0),dim3(64,8)>>>(m,n,a,lda,x,y);
}
} // namespace tiled_gemv
