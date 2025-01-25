/*
 * @file gMatMul.cu
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include "cuAlgo.h"
#include "internals/utils.hpp"

template <
unsigned int BM,
unsigned int BN,
unsigned int BK,
unsigned int TM,
typename T
>
CUALGO_GLOBAL
void gMatMulKernel(T                      alpha,
                   const T * __restrict__ A    ,
                   const T * __restrict__ B    ,
                   T                      beta ,
                   T       * __restrict__ C    ,
                   unsigned int           M    ,
                   unsigned int           N    ,
                   unsigned int           K    ) {

    // output matrix block in this thread block
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    T threadResults[TM] = {0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            T tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            threadResults[resIdx] +=
                As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

namespace cuAlgo {

    template <typename T>
    void gMatMul(T             alpha ,
                 const T      *A     ,
                 const T      *B     ,
                 T             beta  ,
                 T            *C     ,
                 unsigned int  M     ,
                 unsigned int  N     ,
                 unsigned int  K     ,
                 cudaStream_t  stream,
                 bool          async ) {

        static constexpr uint BM = 64;
        static constexpr uint BN = 64;
        static constexpr uint BK = 8;
        static constexpr uint TM = 8;

        // create as many blocks as necessary to map all of C
        dim3 blocksPerGrid(div_ceil(N, BN), div_ceil(M, BM));
        dim3 threadsPerBlock((BM*BN)/TM);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gMatMulKernel<BM,BN,BK,TM, T>),
              alpha, A, B, beta, C, M, N, K);
    }

	void gMatMulInt(int           alpha ,
	                const int    *A     ,
	                const int    *B     ,
	                int           beta  ,
	                int          *C     ,
	                unsigned int  M     ,
	                unsigned int  N     ,
	                unsigned int  K     ,
	                cudaStream_t  stream,
	                bool          async )
	{

		gMatMul<int>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
	}

	void gMatMulFloat(float         alpha ,
	                  const float  *A     ,
	                  const float  *B     ,
	                  float         beta  ,
	                  float        *C     ,
	                  unsigned int  M     ,
	                  unsigned int  N     ,
	                  unsigned int  K     ,
	                  cudaStream_t  stream,
	                  bool          async )
	{

		gMatMul<float>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
	}

	void gMatMulDouble(double        alpha ,
	                   const double *A     ,
	                   const double *B     ,
	                   double        beta  ,
	                   double       *C     ,
	                   unsigned int  M     ,
	                   unsigned int  N     ,
	                   unsigned int  K     ,
	                   cudaStream_t  stream,
	                   bool          async )
	{

		gMatMul<double>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
	}

	template void gMatMul(float , const float  *, const float  *,
	                      float , float  *,
	                      unsigned int, unsigned int, unsigned int,
	                      cudaStream_t, bool);
	template void gMatMul(double, const double *, const double *,
	                      double, double *,
	                      unsigned int, unsigned int, unsigned int,
	                      cudaStream_t, bool);
}
