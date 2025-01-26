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
unsigned int TN,
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

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;

    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const uint strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
    T threadResults[TM * TN] = {0};
    // register caches for As and Bs
    T regM[TM] = {0};
    T regN[TN] = {0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
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

        static constexpr uint BM = 128;
        static constexpr uint BN = 128;
        static constexpr uint BK = 8;
        static constexpr uint TM = 8;
        static constexpr uint TN = 8;

        // create as many blocks as necessary to map all of C
        dim3 blocksPerGrid(div_ceil(N, BN), div_ceil(M, BM));
        dim3 threadsPerBlock((BM*BN)/(TM*TN));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gMatMulKernel<BM,BN,BK,TM,TN, T>),
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
