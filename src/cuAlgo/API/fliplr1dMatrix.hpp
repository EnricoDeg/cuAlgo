/*
 * @file fliplr1dMatrix.hpp
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

template <typename T>
CUALGO_GLOBAL
void fliplrMatrixKernelDim0(T * CUALGO_RESTRICT data,
                            unsigned int mRows,
                            unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int mRowsLim = mRows % 2 == 0 ? mRows / 2 : (mRows - 1) / 2;

    if (tidx < mCols && tidy < mRowsLim) {

        // rows swap
        T tmp = data[( mRows - 1 - tidy) * mCols + tidx];
        data[(mRows - 1 - tidy) * mCols + tidx] = data[tidy * mCols + tidx];
        data[             tidy  * mCols + tidx] = tmp;
    }
}

template <typename T>
CUALGO_GLOBAL
void fliplrMatrixKernelDim1(T * CUALGO_RESTRICT data,
                            unsigned int mRows,
                            unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int mColsLim = mCols % 2 == 0 ? mCols / 2 : (mCols - 1) / 2;

    if (tidx < mColsLim && tidy < mRows) {

        // columns swap
        T tmp = data[tidy * mCols + (mCols - 1 - tidx)];
        data[tidy * mCols + (mCols - 1 - tidx)] = data[tidy * mCols + tidx];
        data[tidy * mCols + tidx            ] = tmp;
    }
}

namespace cuAlgo {

    /**
    * @brief   Flip rows or columns of a matrix in place
    * 
    * @details The `dim` argument defines if rows or columns 
    *          should be flipped.
    * 
    * @param[inout] data  pointer to matrix to be flipped
    * @param[in]    dim   dimension to flip
    * @param[in]    mRows non-contiguous dimension of the matrix
    * @param[in]    mCols contiguous dimension of the matrix
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
    typename T>
    void fliplr1dMatrix(T *data,
                        unsigned int dim,
                        unsigned int mRows,
                        unsigned int mCols,
                        cudaStream_t stream = 0,
                        bool async = false) {

        if (dim == 0) {

            unsigned int mRowsLim = mRows % 2 == 0 ? mRows / 2 : (mRows - 1) / 2;

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(mCols, BlockSizeX),
                               div_ceil(mRowsLim, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 fliplrMatrixKernelDim0<T>,
                 data, mRows, mCols);
        } else if (dim == 1) {

            unsigned int mColsLim = mCols % 2 == 0 ? mCols / 2 : (mCols - 1) / 2;

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(mColsLim, BlockSizeX),
                               div_ceil(mRows, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 fliplrMatrixKernelDim1<T>,
                 data, mRows, mCols);
        }
    }
}
