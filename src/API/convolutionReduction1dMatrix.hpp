/*
 * @file convolutionReduction1dMatrix.hpp
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
#include "cuAlgo.hpp"
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template<
unsigned int BlockSize,
typename T
>
CUALGO_GLOBAL
void convolutionReduction1dMatrixKernel(const T * CUALGO_RESTRICT R,
                                        const T * CUALGO_RESTRICT V,
                                        T * CUALGO_RESTRICT C,
                                        unsigned int N,
                                        unsigned int K,
                                        unsigned int chunks) {

    const unsigned int tid = blockIdx.x * BlockSize + threadIdx.x;

    if (tid > N / 2 * chunks)
        return;

    T tmp1 = 0;
    T tmp2 = 0;

    CUALGO_UNROLL
    for (unsigned int i = 0; i < K / chunks; ++i) {

        unsigned int col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
        const unsigned int row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

        if (col == 0) {

            tmp1 += R[col + N * row] * V[col + N * row];
            tmp2 += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
        } else if (col > 0 && col < N /2) {

            tmp1 += R[col + N * row] * V[col + N * row] -
                    R[N - col + N * row] * V[N - col + N * row] ;
            tmp2 += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
                    R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
        }
    }

    unsigned int tidx = ( tid ) % ( N / 2 ) ;
    unsigned int tidy = ( tid ) / ( N / 2 ) ;
    C[tidx + N * tidy]         = tmp1;
    C[tidx + N / 2 + N * tidy] = tmp2;
}

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T
>
CUALGO_GLOBAL
void convolutionReduction1dMatrixKernel1(const T * CUALGO_RESTRICT R,
                                         const T * CUALGO_RESTRICT V,
                                         T * CUALGO_RESTRICT C,
                                         unsigned int N,
                                         unsigned int K,
                                         unsigned int chunks) {

    CUALGO_SHMEM T sdata1[BlockSizeY][BlockSizeX];
    CUALGO_SHMEM T sdata2[BlockSizeY][BlockSizeY];

    const unsigned int col = blockIdx.x * BlockSizeX + threadIdx.x;
          unsigned int row = blockIdx.y * BlockSizeY + threadIdx.y;

    if (col < N / 2 && row < chunks) {

        sdata1[threadIdx.y][threadIdx.x] = 0;
        sdata2[threadIdx.y][threadIdx.x] = 0;

        if (col == 0) {

            CUALGO_UNROLL
            for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
                sdata1[threadIdx.y][threadIdx.x] += R[col         + N * row] * V[col         + N * row];
                sdata2[threadIdx.y][threadIdx.x] += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
            }
        } else if (col > 0 && col < N / 2) {

            CUALGO_UNROLL
            for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
                sdata1[threadIdx.y][threadIdx.x] += R[col + N * row]         * V[col + N * row] -
                                                    R[N - col + N * row]     * V[N - col + N * row] ;
                sdata2[threadIdx.y][threadIdx.x] += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
                                                    R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
            }
        }
    }

    __syncthreads();

    for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
        if (threadIdx.y < s) {
            sdata1[threadIdx.y][threadIdx.x] += sdata1[threadIdx.y + s][threadIdx.x];
            sdata2[threadIdx.y][threadIdx.x] += sdata2[threadIdx.y + s][threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        C[col         + N * blockIdx.y] = sdata1[0][threadIdx.x];
        C[col + N / 2 + N * blockIdx.y] = sdata2[0][threadIdx.x];
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform 1D convolution on the input matrices and then a 
    *          1D reduction in the slow dimension.
    * 
    * @details This function combines convolution1dMatrix() and reduction1dMatrix()
    *          in a single kernel. The input matrices has dimensions {N,K}
    *          and the output vector has dimension {N}.
    * 
    * @param[in]  R pointer to the first input matrix for the convolution.
    *               The signals are assumed to be in the frequency domain already.
    *               The matrix has dimensions {N,K}.
    * @param[in]  V pointer to the second input matrix for the convolution.
    *               The signals are assumed to be in the frequency domain already.
    *               The matrix has dimensions {N,K}.
    * @param[out] C pointer to the the output vector with the result of the 
    *               convolution and the reduction.
    *               The signals are still in the frequency domain already.
    *               The vector has dimension {N}.
    * @param[in]  N contiguous dimension of the input matrices
    * @param[in]  K non-contiguous dimension of the input matrices
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int ItemsPerThread,
    typename T>
    void convolutionReduction1dMatrix(T *R,
                                      T *V,
                                      T *C,
                                      unsigned int N,
                                      unsigned int K,
                                      cudaStream_t stream = 0,
                                      bool async = false) {

        unsigned int chunks = K / ItemsPerThread;
        static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY;

        if (chunks > BlockSizeY) {

            T * d_buffer;
            check_cuda( cudaMalloc(&d_buffer, N * chunks / BlockSizeY * sizeof(T)) );

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(N / 2, BlockSizeX),
                               div_ceil(chunks, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(convolutionReduction1dMatrixKernel1<BlockSizeX, BlockSizeY, T>),
                 R, V, d_buffer, N, K, chunks);

            reduction1dMatrix<T>(d_buffer, C, N, chunks / BlockSizeY, stream, async);

            check_cuda( cudaFree ( d_buffer ) );
        } else if (chunks < BlockSizeY && chunks > 1) {

            T * d_buffer;
            check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(T)) );

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(div_ceil(N / 2 * chunks, BlockSize));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(convolutionReduction1dMatrixKernel<BlockSize, T>),
                 R, V, d_buffer, N, K, chunks);

            reduction1dMatrix<T>(d_buffer, C, N, chunks, stream, async);

            check_cuda( cudaFree ( d_buffer ) );
        } else {

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(div_ceil(N, BlockSize));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(convolutionReduction1dMatrixKernel<BlockSize, T>),
                 R, V, C, N, K, chunks);
        }
    }
}
