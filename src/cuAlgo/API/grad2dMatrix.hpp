/*
 * @file grad2dMatrix.hpp
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

template<
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void gradMatrixKernel(const T * CUALGO_RESTRICT A,
                      T * CUALGO_RESTRICT Ax,
                      T * CUALGO_RESTRICT Ay,
                      unsigned int M,
                      unsigned int N) {

    CUALGO_SHMEM T sdata[BlockSizeY][BlockSizeX];

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {

        sdata[threadIdx.y][threadIdx.x] = A[x + y * M];

        if (y == 0)
            Ax[x + y * M] = sdata[threadIdx.y][threadIdx.x];
        else
            Ax[x + y * M] = sdata[threadIdx.y][threadIdx.x] - A[x + (y - 1) * M];

        if (x == 0)
            Ay[x + y * M] = sdata[threadIdx.y][threadIdx.x];
        else
            Ay[x + y * M] = sdata[threadIdx.y][threadIdx.x] - A[x - 1 + y * M];
    }
}

namespace cuAlgo {

    /**
    * @brief   Compute the matrix gradient
    * 
    * @details The gradient is computed using first order
    *          accuracy.
    * 
    * @param[in]  A input matrix of size {N,M}
    * @param[out] Ax output matrix with x derivative.
    *             The matrix has the same size of A.
    * @param[out] Ay output matrix with y derivative.
    *             The matrix has the same size of A.
    * @param[in]  M size of contiguous dimension
    * @param[in]  N size of non-contiguous dimension
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
    typename T>
    void grad2dMatrix(T *A,
                      T *Ax,
                      T *Ay,
                      unsigned int M,
                      unsigned int N,
                      cudaStream_t stream = 0,
                      bool async = false) {

        dim3 blocksPerGrid(div_ceil(M, BlockSizeX), div_ceil(N, BlockSizeY));
        dim3 threadsPerBlock(BlockSizeX , BlockSizeY);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gradMatrixKernel<BlockSizeX, BlockSizeY, T>),
              A, Ax, Ay, M, N);
    }
}
