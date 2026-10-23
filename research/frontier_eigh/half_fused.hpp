#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

// Deliberately bounded prototype: one wave owns a 16x16 output tile; K tiles
// are 16 wide. rocWMMA owns all lane/fragment mappings. No global half buffers,
// atomics, split-K, or undocumented MFMA lane assumptions. Two FP32 register
// accumulators retain HH and both cross products until a single C epilogue.
// The simple tile duplicates operand loading between blocks; this is a diagnostic
// candidate, not an assertion that it can match tuned rocBLAS occupancy/reuse.
namespace half_fused {
__device__ inline void split(float x,rocwmma::float16_t& hi,rocwmma::float16_t& lo) {
    __half h=__float2half(fabsf(x)<0x1p-14f?0.f:x);
    hi=rocwmma::float16_t(__half2float(h));
    lo=rocwmma::float16_t(__half2float(__float2half((x-__half2float(h))*4096.f)));
}

template<bool TA,bool TB>
__global__ __launch_bounds__(64) void gemm(int m,int n,int k,float alpha,
    const float* a,int lda,const float* b,int ldb,float beta,float* c,int ldc) {
    using namespace rocwmma;
    __shared__ __align__(32) float16_t ah[256],al[256],bh[256],bl[256];
    __shared__ __align__(32) float hh_tile[256],cross_tile[256];
    using A=fragment<matrix_a,16,16,16,float16_t,col_major>;
    using B=fragment<matrix_b,16,16,16,float16_t,col_major>;
    using C=fragment<accumulator,16,16,16,float>;
    C hh,cross;
    fill_fragment(hh,0.f); fill_fragment(cross,0.f);
    const int row0=blockIdx.x*16,col0=blockIdx.y*16;
    for(int kk=0;kk<k;kk+=16) {
        // All threads participate, including edge tiles. Zero-fill partial K/M/N.
        for(int z=threadIdx.x;z<256;z+=64) {
            int row=row0+z%16,ka=kk+z/16;
            int kb=kk+z%16,col=col0+z/16;
            float av=row<m && ka<k?a[TA?ka+size_t(row)*lda:row+size_t(ka)*lda]:0.f;
            float bv=col<n && kb<k?b[TB?col+size_t(kb)*ldb:kb+size_t(col)*ldb]:0.f;
            split(av,ah[z],al[z]); split(bv,bh[z],bl[z]);
        }
        synchronize_workgroup();
        A a_hi,a_lo; B b_hi,b_lo;
        load_matrix_sync(a_hi,ah,16); load_matrix_sync(a_lo,al,16);
        load_matrix_sync(b_hi,bh,16); load_matrix_sync(b_lo,bl,16);
        mma_sync(hh,a_hi,b_hi,hh);
        mma_sync(cross,a_hi,b_lo,cross);
        mma_sync(cross,a_lo,b_hi,cross);
        synchronize_workgroup(); // No LDS overwrite before all fragment reads finish.
    }
    store_matrix_sync(hh_tile,hh,16,mem_col_major);
    store_matrix_sync(cross_tile,cross,16,mem_col_major);
    synchronize_workgroup();
    for(int z=threadIdx.x;z<256;z+=64) {
        int row=row0+z%16,col=col0+z/16;
        if(row<m && col<n) {
            size_t offset=row+size_t(col)*ldc;
            float result=alpha*(hh_tile[z]+cross_tile[z]/4096.f);
            // beta==0 must not read old/uninitialized C.
            c[offset]=beta==0.f?result:result+beta*c[offset];
        }
    }
}

inline void launch(bool ta,bool tb,int m,int n,int k,float alpha,
    const float* a,int lda,const float* b,int ldb,float beta,float* c,int ldc) {
    dim3 grid((m+15)/16,(n+15)/16);
    if(ta && tb) gemm<true,true><<<grid,64>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else if(ta) gemm<true,false><<<grid,64>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else if(tb) gemm<false,true><<<grid,64>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else gemm<false,false><<<grid,64>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

// Four-wave CTA: share each converted 64x16 A / 16x64 B tile among all waves.
// Each wave owns a 32x32 quadrant via four 16x16 fragments. Operand LDS is
// 8 KiB; its storage is reused by the 16 KiB FP32 epilogue tile after K ends.
template<bool TA,bool TB>
__global__ __launch_bounds__(256) void gemm_tile(int m,int n,int k,float alpha,
    const float* a,int lda,const float* b,int ldb,float beta,float* c,int ldc) {
    using namespace rocwmma;
    __shared__ __align__(32) union {
        struct { float16_t ah[1024],al[1024],bh[1024],bl[1024]; } inputs;
        float output[4096];
    } tile;
    using A=fragment<matrix_a,16,16,16,float16_t,col_major>;
    using B=fragment<matrix_b,16,16,16,float16_t,col_major>;
    using C=fragment<accumulator,16,16,16,float>;
    C hh[2][2],cross[2][2];
    #pragma unroll
    for(int i=0;i<2;++i) {
        #pragma unroll
        for(int j=0;j<2;++j) { fill_fragment(hh[i][j],0.f); fill_fragment(cross[i][j],0.f); }
    }
    const int wave=threadIdx.x/64,wr=(wave%2)*32,wc=(wave/2)*32;
    const int row0=blockIdx.x*64,col0=blockIdx.y*64;
    for(int kk=0;kk<k;kk+=16) {
        for(int z=threadIdx.x;z<1024;z+=256) {
            // Traverse the physical contiguous dimension of each source, then
            // transpose into the fixed col-major LDS layouts when necessary.
            int az=TA?z/16+(z%16)*64:z;
            int bz=TB?z/64+(z%64)*16:z;
            int row=row0+az%64,ka=kk+az/64;
            int kb=kk+bz%16,col=col0+bz/16;
            float av=row<m && ka<k?a[TA?ka+size_t(row)*lda:row+size_t(ka)*lda]:0.f;
            float bv=col<n && kb<k?b[TB?col+size_t(kb)*ldb:kb+size_t(col)*ldb]:0.f;
            split(av,tile.inputs.ah[az],tile.inputs.al[az]);
            split(bv,tile.inputs.bh[bz],tile.inputs.bl[bz]);
        }
        synchronize_workgroup();
        A a_hi[2],a_lo[2]; B b_hi[2],b_lo[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            load_matrix_sync(a_hi[i],tile.inputs.ah+wr+i*16,64);
            load_matrix_sync(a_lo[i],tile.inputs.al+wr+i*16,64);
            load_matrix_sync(b_hi[i],tile.inputs.bh+(wc+i*16)*16,16);
            load_matrix_sync(b_lo[i],tile.inputs.bl+(wc+i*16)*16,16);
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            #pragma unroll
            for(int j=0;j<2;++j) {
                mma_sync(hh[i][j],a_hi[i],b_hi[j],hh[i][j]);
                mma_sync(cross[i][j],a_hi[i],b_lo[j],cross[i][j]);
                mma_sync(cross[i][j],a_lo[i],b_hi[j],cross[i][j]);
            }
        }
        synchronize_workgroup(); // All four waves finish before staging the next K tile.
    }
    #pragma unroll
    for(int i=0;i<2;++i) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            // Identical accumulator types have identical element ownership; only
            // elementwise arithmetic, never a hard-coded lane-to-matrix mapping.
            #pragma unroll
            for(unsigned z=0;z<C::num_elements;++z) hh[i][j][z]+=cross[i][j][z]/4096.f;
            store_matrix_sync(tile.output+wr+i*16+(wc+j*16)*64,hh[i][j],64,mem_col_major);
        }
    }
    synchronize_workgroup();
    for(int z=threadIdx.x;z<4096;z+=256) {
        int row=row0+z%64,col=col0+z/64;
        if(row<m && col<n) {
            size_t offset=row+size_t(col)*ldc;
            float result=alpha*tile.output[z];
            c[offset]=beta==0.f?result:result+beta*c[offset];
        }
    }
}

inline void launch_tile(bool ta,bool tb,int m,int n,int k,float alpha,
    const float* a,int lda,const float* b,int ldb,float beta,float* c,int ldc) {
    dim3 grid((m+63)/64,(n+63)/64);
    if(ta && tb) gemm_tile<true,true><<<grid,256>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else if(ta) gemm_tile<true,false><<<grid,256>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else if(tb) gemm_tile<false,true><<<grid,256>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else gemm_tile<false,false><<<grid,256>>>(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
} // namespace half_fused
