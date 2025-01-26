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

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 1 + 0) * Ncol + row + offset] = x;
    }
};

template <typename T>
struct vecT<T, 2>
{
    T x;
    T y;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 2 + 0) * Ncol + row + offset] = x;
        As[(col * 2 + 1) * Ncol + row + offset] = y;
    }
};

template <typename T>
struct vecT<T, 3>
{
    T x;
    T y;
    T z;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 3 + 0) * Ncol + row + offset] = x;
        As[(col * 3 + 1) * Ncol + row + offset] = y;
        As[(col * 3 + 2) * Ncol + row + offset] = z;
    }
};

template <typename T>
struct vecT<T, 4>
{
    T x;
    T y;
    T z;
    T w;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
        w = alpha * results[3] + beta * w;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 4 + 0) * Ncol + row + offset] = x;
        As[(col * 4 + 1) * Ncol + row + offset] = y;
        As[(col * 4 + 2) * Ncol + row + offset] = z;
        As[(col * 4 + 3) * Ncol + row + offset] = w;
    }
};

template<
unsigned int WITER,
unsigned int TSIZE,
unsigned int WSUBCOL,
typename T
>
CUALGO_DEVICE
CUALGO_FORCE_INLINE
void fill_register(T * reg, T* shmem) {

    for (unsigned int wSubIdx = 0; wSubIdx < WITER; ++wSubIdx) {
        for (unsigned int i = 0; i < TSIZE; ++i) {
            reg[wSubIdx * TSIZE + i] =
                shmem[wSubIdx * WSUBCOL + i];
        }
    }
}

template <
unsigned int BM,
unsigned int BN,
unsigned int BK,
unsigned int WM,
unsigned int WN,
unsigned int WNITER,
unsigned int TM,
unsigned int TN,
unsigned int NUM_THREADS,
typename T
>
CUALGO_GLOBAL
CUALGO_LAUNCH_BOUNDS(NUM_THREADS)
void gMatMulKernel(T alpha,
                   T * CUALGO_RESTRICT A,
                   T * CUALGO_RESTRICT B,
                   T beta,
                   T * CUALGO_RESTRICT C,
                   unsigned int M,
                   unsigned int N,
                   unsigned int K)
{
    // load 128 bit at once
    static constexpr int numberVectorElements = 16 / sizeof(T);
    using vecN = vecT<T,numberVectorElements>;

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const unsigned int warpIdx = threadIdx.x / CUALGO_WARPSIZE;
    const unsigned int warpCol = warpIdx % (BN / WN);
    const unsigned int warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr unsigned int WMITER = (WM * WN) / (CUALGO_WARPSIZE * TM * TN * WNITER);
    constexpr unsigned int WSUBM = WM / WMITER; // 64/2=32
    constexpr unsigned int WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const unsigned int threadIdxInWarp = threadIdx.x % CUALGO_WARPSIZE;
    const unsigned int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const unsigned int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // allocate space for the current blocktile in smem
    CUALGO_SHMEM T shmem[BM * BK + BK * BN];
    T * As = shmem;
    T * Bs = shmem + BM * BK;

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const unsigned int innerRowA = threadIdx.x / (BK / numberVectorElements);
    const unsigned int innerColA = threadIdx.x % (BK / numberVectorElements);
    constexpr unsigned int rowStrideA = (NUM_THREADS * numberVectorElements) / BK;
    const unsigned int innerRowB = threadIdx.x / (BN / numberVectorElements);
    const unsigned int innerColB = threadIdx.x % (BN / numberVectorElements);
    constexpr unsigned int rowStrideB = NUM_THREADS / (BN / numberVectorElements);

    // allocate thread-local cache for results in registerfile
    T threadResults[WMITER * TM * WNITER * TN] = {0};
    // register caches for As and Bs
    T regM[WMITER * TM] = {0};
    T regN[WNITER * TN] = {0};

    // outer-most loop over block tiles
    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // load GMEM -> SMEM
        // transpose A while loading it
        for (unsigned int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            vecN tmp = reinterpret_cast<vecN *>(
                &A[(innerRowA + offset) * K + innerColA * numberVectorElements])[0];
            tmp.load_transpose(As, innerRowA, innerColA, offset, BM);
        }

        for (unsigned int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<vecN *>(
                &Bs[(innerRowB + offset) * BN + innerColB * numberVectorElements])[0] =
                    reinterpret_cast<vecN *>(
                    &B[(innerRowB + offset) * N + innerColB * numberVectorElements])[0];
        }
        __syncthreads();

        // calculate per-thread results
        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // SHMEM -> registers
            fill_register<WMITER, TM, WSUBM>(regM,
                &As[(dotIdx * BM) + warpRow * WM + threadRowInWarp * TM]);
            fill_register<WNITER, TN, WSUBN>(regN,
                &Bs[(dotIdx * BN) + warpCol * WN + threadColInWarp * TN]);

            // Core computation
            for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                                    regM[wSubRowIdx * TM + resIdxM] *
                                    regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            T *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += numberVectorElements) {
                    // load C vector into registers
                    vecN tmp = reinterpret_cast<vecN *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.store(threadResults + i, alpha, beta);
                    // write back
                    reinterpret_cast<vecN *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
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
        static constexpr uint BK = 16;
        static constexpr uint WN = 64;
        static constexpr uint WM = 64;
        static constexpr uint WNITER = 4;
        static constexpr uint TM = 8;
        static constexpr uint TN = 4;
        static constexpr uint NUM_THREADS = 128;

        // create as many blocks as necessary to map all of C
        dim3 blocksPerGrid(div_ceil(N, BN), div_ceil(M, BM));
        dim3 threadsPerBlock(NUM_THREADS);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gMatMulKernel<BM,BN,BK,WM, WN, WNITER, TM, TN, NUM_THREADS, T>),
              alpha, A, B, beta, C, M, N, K);
    }
}
