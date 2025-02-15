/*
 * @file convolutionTaperReduction1dMatrix.hpp
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
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/kernelParameters.hpp"
#include "cuAlgo/API/reduction1dMatrix.hpp"

template<
unsigned int BlockSize,
typename T
>
CUALGO_GLOBAL
void convolutionTaperReduction1dMatrixKernel(const T * CUALGO_RESTRICT R,
                                             const T * CUALGO_RESTRICT V,
                                             const T * CUALGO_RESTRICT Taper,
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

            tmp1 += Taper[row] * ( R[col + N * row] * V[col + N * row] );
            tmp2 += Taper[row] * ( R[col + N / 2 + N * row] * V[col + N / 2 + N * row] );
        } else if (col > 0 && col < N /2) {

            tmp1 += Taper[row] * ( R[col + N * row] * V[col + N * row] -
                               R[N - col + N * row] * V[N - col + N * row] );
            tmp2 += Taper[row] * ( R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
                               R[col + N / 2 + N * row] * V[N / 2 - col + N * row] );
        }
    }

    unsigned int tidx = ( tid ) % ( N / 2 ) ;
    unsigned int tidy = ( tid ) / ( N / 2 ) ;
    C[tidx + N * tidy]         = tmp1;
    C[tidx + N / 2 + N * tidy] = tmp2;
}

namespace cuAlgo {

    /**
    * @brief   Perform 1D convolution on the input matrices, then apply a taper
    *          on the slow dimension and finally perform a 1D reduction in the
    *          slow dimension.
    * 
    * @details This function is similar to convolutionReduction1dMatrix() but a
    *          taper defined in the slow dimension is applied before the 1D
    *          reduction.
    * 
    * @param[in]  R     pointer to the first input matrix for the convolution.
    *                   The signals are assumed to be in the frequency domain already.
    *                   The matrix has dimensions {N,K}.
    * @param[in]  V     pointer to the second input matrix for the convolution.
    *                   The signals are assumed to be in the frequency domain already.
    *                   The matrix has dimensions {N,K}.
    * @param[in]  Taper pointer to the input vector with the taper values.
    *                   The vector is defined in the slow dimension of the input
    *                   matrices so it has dimension {K}.
    * @param[out] C     pointer to the the output vector with the result of the
    *                   convolution and the reduction.
    *                   The signals are still in the frequency domain already.
    *                   The vector has dimension {N}.
    * @param[in]  N     contiguous dimension of the input matrices
    * @param[in]  K     non-contiguous dimension of the input matrices
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
    typename T
    >
    void convolutionTaperReduction1dMatrix(T *R,
                                           T *V,
                                           T *Taper,
                                           T *C,
                                           unsigned int N,
                                           unsigned int K,
                                           cudaStream_t stream = 0,
                                           bool async = false) {

        static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY;
        unsigned int chunks = K / ItemsPerThread;

        if (chunks > 1) {

            T * d_buffer;
            check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(T)) );

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(div_ceil(N / 2 * chunks, BlockSize));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 CUALGO_KERNEL_NAME(convolutionTaperReduction1dMatrixKernel<BlockSize,T>),
                 R, V, Taper, d_buffer, N, K, chunks);

            reduction1dMatrix<BlockSizeX, BlockSizeY, T>(d_buffer, C, N, chunks, 0, false);

            check_cuda( cudaFree ( d_buffer ) );
        } else {

            dim3 threadsPerBlock(THREADS_PER_BLOCK);
            dim3 blocksPerGrid(div_ceil(N, THREADS_PER_BLOCK));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 convolutionTaperReduction1dMatrixKernel<T>,
                 R, V, Taper, C, N, K, chunks);
        }
    }
}
