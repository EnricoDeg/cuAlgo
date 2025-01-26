/*
 * @file gMatMul.hpp
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

    __device__
    __forceinline__
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
    }

    __device__
    __forceinline__
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int Ncol) {
        As[(col * 1 + 0) * Ncol + row] = x;
    }
};

template <typename T>
struct vecT<T, 2>
{
    T x;
    T y;

    __device__
    __forceinline__
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
    }

    __device__
    __forceinline__
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int Ncol) {
        As[(col * 2 + 0) * Ncol + row] = x;
        As[(col * 2 + 1) * Ncol + row] = y;
    }
};

template <typename T>
struct vecT<T, 3>
{
    T x;
    T y;
    T z;

    __device__
    __forceinline__
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
    }

    __device__
    __forceinline__
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int Ncol) {
        As[(col * 3 + 0) * Ncol + row] = x;
        As[(col * 3 + 1) * Ncol + row] = y;
        As[(col * 3 + 2) * Ncol + row] = z;
    }
};

template <typename T>
struct vecT<T, 4>
{
    T x;
    T y;
    T z;
    T w;

    __device__
    __forceinline__
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
        w = alpha * results[3] + beta * w;
    }

    __device__
    __forceinline__
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int Ncol) {
        As[(col * 4 + 0) * Ncol + row] = x;
        As[(col * 4 + 1) * Ncol + row] = y;
        As[(col * 4 + 2) * Ncol + row] = z;
        As[(col * 4 + 3) * Ncol + row] = w;
    }
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
            reinterpret_cast<vecN *>(&A[innerRowA * K + innerColA * numberVectorElements])[0];
        tmp.load_transpose(As, innerRowA, innerColA, BM);

        reinterpret_cast<vecN *>(&Bs[innerRowB * BN + innerColB * numberVectorElements])[0] =
            reinterpret_cast<vecN *>(&B[innerRowB * N + innerColB * numberVectorElements])[0];
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
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += numberVectorElements) {
            // load C vector into registers
            vecN tmp = reinterpret_cast<vecN *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            // perform GEMM update in reg
            tmp.store(&threadResults[resIdxM * TN + resIdxN], alpha, beta);
            // write back
            reinterpret_cast<vecN *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform general matrix-matrix multiplication
    * 
    * @details The following operation is performed
    *          C = alpha * A * B + beta * C
    * 
    * @param[in]    A     pointer to the input matrix.
    *                     The matrix has dimensions {K,M}.
    * @param[in]    B     pointer to the input matrix.
    *                     The matrix has dimensions {N,K}.
    * @param[inout] C     pointer to the output matrix.
    *                     The matrix has dimensions {N,M}.
    * @param[in]    M     non-contiguous dimension of the A and C matrices
    * @param[in]    N     contiguous dimension of the B and C matrix
    * @param[in]    K     contiguous dimension of the A matrix
    *                     non-contiguous dimension of the B matrix
    * @param[in]    alpha scalar parameter to apply to A * B
    * @param[in]    beta  scalar parameter to apply to C
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <typename T>
    void gMatMul(T             alpha ,
                 T      *A     ,
                 T      *B     ,
                 T             beta  ,
                 T            *C     ,
                 unsigned int  M     ,
                 unsigned int  N     ,
                 unsigned int  K     ,
                 cudaStream_t  stream = 0,
                 bool          async = false) {

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
}
