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

template <typename T, int vec_elements>
struct vecT
{
  static_assert(!sizeof(T), "cuAlgo can only have 1-4 elements");
};

template <typename T>
struct vecT<T, 1>
{
  T x;
};

template <typename T>
struct vecT<T, 2>
{
  T x;
  T y;
};

template <typename T>
struct vecT<T, 3>
{
  T x;
  T y;
  T z;
};

template <typename T>
struct vecT<T, 4>
{
  T x;
  T y;
  T z;
  T w;
};

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
                   T * __restrict__ A    ,
                   T * __restrict__ B    ,
                   T                      beta ,
                   T       * __restrict__ C    ,
                   unsigned int           M    ,
                   unsigned int           N    ,
                   unsigned int           K    )
{
    static constexpr int numberVectorElements = 16 / sizeof(T);
    using vecN = vecT<T,numberVectorElements>;
    static_assert(numberVectorElements == 4, "must be vector of 4 for now");

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

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
    uint innerRowA = threadIdx.x / (BK / numberVectorElements);
    uint innerColA = threadIdx.x % (BK / numberVectorElements);
    uint innerRowB = threadIdx.x / (BN / numberVectorElements);
    uint innerColB = threadIdx.x % (BN / numberVectorElements);

    // allocate thread-local cache for results in registerfile
    T threadResults[TM * TN] = {0};
    // register caches for As and Bs
    T regM[TM] = {0};
    T regN[TN] = {0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // populate the SMEM caches
        // transpose A while loading it
        vecN tmp =
            reinterpret_cast<vecN *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<vecN *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<vecN *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // load C vector into registers
            vecN tmp = reinterpret_cast<vecN *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];

            // perform GEMM update in reg
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
            // write back
            reinterpret_cast<vecN *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}

namespace cuAlgo {

    template <typename T>
    void gMatMul(T             alpha ,
                 T      *A     ,
                 T      *B     ,
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
	                int    *A     ,
	                int    *B     ,
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
}
