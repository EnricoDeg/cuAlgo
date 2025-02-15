/*
 * @file gConvolutionCorrelation1dMatrix.hpp
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

#ifndef GCONVOLUTIONCORRELATION1DMATRIX_H
#define GCONVOLUTIONCORRELATION1DMATRIX_H

#include <cuda.h>
#include "cuAlgo/internals/operations.hpp"
#include "cuAlgo/internals/templateShMem.hpp"
#include "cuAlgo/internals/utils.hpp"

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T,
template<typename> class op_t>
CUALGO_GLOBAL
void gConvolutionCorrelation1dMatrixKernel(const T * CUALGO_RESTRICT R,
                                           const T * CUALGO_RESTRICT V,
                                           T * CUALGO_RESTRICT C,
                                           unsigned int N,
                                           unsigned int K,
                                           unsigned int chunks,
                                           op_t<T> Op) {

    const unsigned int col = blockIdx.x * BlockSizeX + threadIdx.x;
          unsigned int row = blockIdx.y * BlockSizeY + threadIdx.y;

    if (col < N / 2 && row < chunks) {

        if (col == 0) {

            CUALGO_UNROLL
            for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
                C[col         + N * row] = Op.firstRealImag(&R[col + N * row], &V[col + N * row]);
                C[col + N / 2 + N * row] = Op.firstRealImag(&R[col + N / 2 + N * row], &V[col + N / 2 + N * row]);
            }
        } else if (col > 0 && col < N / 2) {

            CUALGO_UNROLL
            for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
                C[col         + N * row] = Op.nReal(&R[    col + N * row], &V[    col + N * row],
                                                    &R[N - col + N * row], &V[N - col + N * row]) ;
                C[col + N / 2 + N * row] = Op.nImag(&R[N / 2 - col + N * row], &V[N / 2 + col + N * row],
                                                    &R[N / 2 + col + N * row], &V[N / 2 - col + N * row]) ;
            }
        }
    }
}

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
unsigned int ItemsPerThread,
typename T,
template<typename> class op_t>
void gConvolutionCorrelation1dMatrix(T            *R     ,
                                     T            *V     ,
                                     T            *C     ,
                                     unsigned int  N     ,
                                     unsigned int  K     ,
                                     cudaStream_t  stream,
                                     bool          async ) {

    op_t<T> op;

    unsigned int chunks = K / ItemsPerThread;

    dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
    dim3 blocksPerGrid(div_ceil(N / 2, BlockSizeX), div_ceil(chunks, BlockSizeY));
    print_kernel_config(threadsPerBlock, blocksPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
         CUALGO_KERNEL_NAME(gConvolutionCorrelation1dMatrixKernel<BlockSizeX, BlockSizeY, T,op_t>),
         R, V, C, N, K, chunks, op);
}

#endif
