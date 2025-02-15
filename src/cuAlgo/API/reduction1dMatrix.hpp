/*
 * @file reduction1dmatrix.hpp
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
#include <cmath>
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/kernelParameters.hpp"

#define COMPUTE_PER_THREAD 16

template <
unsigned int BlockSize,
typename T>
CUALGO_GLOBAL
void reduction1dMatrixKernel(const T * CUALGO_RESTRICT B,
                             T * CUALGO_RESTRICT C,
                             unsigned int N,
                             unsigned int K,
                             unsigned int chunks) {

    const unsigned int tid = blockIdx.x * BlockSize + threadIdx.x;
    if (tid > N * chunks)
        return;

    const unsigned int col   = tid % N;
    const unsigned int chunk = tid / N;
    const unsigned int tidm = col + chunk * N;

    T tmp = 0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K / chunks; ++i) {
        tmp += B[i * N*chunks + tidm];
    }
    C[col+chunk*N] = tmp;
}

template <
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void reduction1dMatrixKernel1(const T * CUALGO_RESTRICT B,
                              T * CUALGO_RESTRICT C,
                              unsigned int N,
                              unsigned int K,
                              unsigned int chunks) {

    const unsigned int tidx = blockIdx.x * BlockSizeX + threadIdx.x;
    const unsigned int tidy = blockIdx.y * BlockSizeY + threadIdx.y;
    if (tidx + N * tidy > N * chunks)
        return;

    const unsigned int tidm = tidx + tidy * N;

    T tmp = 0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K / chunks; ++i) {
        tmp += B[i * N*chunks + tidm];
    }
    C[tidx+tidy*N] = tmp;
}

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void reduction1dMatrixKernel2(const T * CUALGO_RESTRICT B,
                              T * CUALGO_RESTRICT C,
                              unsigned int N,
                              unsigned int K,
                              unsigned int chunks) {

    CUALGO_SHMEM T sdata[BlockSizeY][BlockSizeX];

    const unsigned int tidx = blockIdx.x * BlockSizeX + threadIdx.x;
    const unsigned int tidy = blockIdx.y * BlockSizeY + threadIdx.y;
    if (tidx + N * tidy > N * chunks)
        return;

    const unsigned int tidm = tidx + tidy * N;

    sdata[threadIdx.y][threadIdx.x] = 0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K / chunks; ++i) {
        sdata[threadIdx.y][threadIdx.x] += B[i * N*chunks + tidm];
    }
    __syncthreads();

    for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
        if (threadIdx.y < s) {
            sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y + s][threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0)
        C[tidx+blockIdx.y*N] = sdata[0][threadIdx.x];
}

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void reduction1dMatrixKernel3(const T * CUALGO_RESTRICT B,
                              T * CUALGO_RESTRICT C,
                              unsigned int N,
                              unsigned int K,
                              unsigned int chunks) {

    CUALGO_SHMEM T sdata[BlockSizeY][BlockSizeX];

    const unsigned int tidx = blockIdx.x * BlockSizeX + threadIdx.x;
    const unsigned int tidy = blockIdx.y * BlockSizeY + threadIdx.y;
    if (tidx + N * tidy > N * chunks)
        return;

    const unsigned int tidm = tidx + tidy * N;

    sdata[threadIdx.y][threadIdx.x] = 0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K / chunks; ++i) {
        sdata[threadIdx.y][threadIdx.x] += B[i * N*chunks + tidm];
    }
    __syncthreads();

    if (threadIdx.y < 16)
        sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y + 16][threadIdx.x];
    __syncthreads();

    if (threadIdx.y <  8)
        sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y +  8][threadIdx.x];
    __syncthreads();

    if (threadIdx.y <  4)
        sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y +  4][threadIdx.x];
    __syncthreads();

    if (threadIdx.y <  2)
        sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y +  2][threadIdx.x];
    __syncthreads();

    if (threadIdx.y <  1)
        sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y +  1][threadIdx.x];
    __syncthreads();

    if (threadIdx.y == 0)
        C[tidx+blockIdx.y*N] = sdata[0][threadIdx.x];
}

namespace cuAlgo {

    /**
    * @brief   Perform 1D reduction on a 2D array (matrix)
    *          of size {N,K}
    * 
    * @details The reduction is done on the slow dimension, so the output
    *          vector has size {N}.
    * 
    * @param[in]  B pointer to input matrix to be reduced
    * @param[out] C pointer to output vector with result of the reduction
    * @param[in]  N contiguous dimension of the input matrix
    * @param[in]  K non-contiguous dimension of the input matrix
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int ItemsPerThread,
    typename T>
    void reduction1dMatrix(T *B,
                           T *C,
                           unsigned int N,
                           unsigned int K,
                           cudaStream_t stream = 0,
                           bool async = false) {

        static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY;
        unsigned int chunks = K / ItemsPerThread;

        if (chunks > BlockSizeY) {

            T * d_buffer;
            check_cuda( cudaMalloc(&d_buffer, N * chunks / 32 *sizeof(T)) );

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(N, BlockSizeX),
                               div_ceil(chunks, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(reduction1dMatrixKernel3<BlockSizeX, BlockSizeY, T>),
                 B, d_buffer, N, K, chunks);

            reduction1dMatrix<BlockSizeX, BlockSizeY, ItemsPerThread, T>(d_buffer, C, N, chunks/32, stream, async);

            check_cuda( cudaFree ( d_buffer ) );
        } else if (chunks < BlockSizeY && chunks > 1) {

            T * d_buffer;
            check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(T)) );

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(div_ceil(N, BlockSize)*chunks);
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(reduction1dMatrixKernel<BlockSize, T>),
                 B, d_buffer, N, K, chunks);

            reduction1dMatrix<BlockSizeX, BlockSizeY, ItemsPerThread, T>(d_buffer, C, N, chunks, stream, async);

            check_cuda( cudaFree ( d_buffer ) );
        } else {

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(div_ceil(N, BlockSize));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(reduction1dMatrixKernel<BlockSize, T>),
                 B, C, N, K, 1);
        }
    }
}
