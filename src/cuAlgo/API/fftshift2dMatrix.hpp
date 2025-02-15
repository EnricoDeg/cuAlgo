/*
 * @file fftshift2dMatrix.hpp
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
#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"

template <typename T>
CUALGO_GLOBAL
void fftshiftMatrixDim0KernelEven(T * CUALGO_RESTRICT data,
                                  unsigned int mRows,
                                  unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx < mCols && tidy < mRows / 2) {

        T tmp = data[(tidy + mRows / 2) * mCols + tidx];
        data[(tidy + mRows / 2) * mCols + tidx] = data[tidy * mCols + tidx];
        data[ tidy * mCols + tidx             ] = tmp;
    }
}

template <typename T>
CUALGO_GLOBAL
void fftshiftMatrixDim1KernelEven(T * CUALGO_RESTRICT data,
                                  unsigned int mRows,
                                  unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx < mCols / 2 && tidy < mRows) {

        T tmp = data[tidy * mCols + tidx + mCols / 2];
        data[tidy * mCols + tidx + mCols / 2] = data[tidy * mCols + tidx];
        data[tidy * mCols + tidx            ] = tmp;
    }
}

template <typename T>
CUALGO_GLOBAL
void fftshiftMatrixKernelEvenEven(T * CUALGO_RESTRICT data,
                                  unsigned int mRows,
                                  unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx < mCols / 2 && tidy < mRows / 2) {

        // first columns swap
        T tmp = data[tidy * mCols + tidx + mCols / 2];
        data[tidy * mCols + tidx + mCols / 2] = data[tidy * mCols + tidx];
        data[tidy * mCols + tidx            ] = tmp;

        // second columns swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx + mCols / 2];
        data[(tidy + mRows / 2) * mCols + tidx + mCols / 2] = data[(tidy + mRows / 2) * mCols + tidx];
        data[(tidy + mRows / 2) * mCols + tidx            ] = tmp;

        // first rows swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx];
        data[(tidy + mRows / 2) * mCols + tidx] = data[tidy * mCols + tidx];
        data[ tidy * mCols + tidx            ] = tmp;

        // second rows swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx + mCols / 2];
        data[(tidy + mRows / 2) * mCols + tidx + mCols / 2] = data[tidy * mCols + tidx + mCols / 2];
        data[ tidy * mCols              + tidx + mCols / 2] = tmp;
    }
}

template <typename T>
CUALGO_GLOBAL
void fftshiftMatrixKernelEvenOddFirstStep(T * CUALGO_RESTRICT data,
                                          unsigned int mRows,
                                          unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx < (mCols - 1) / 2 && tidy < mRows / 2) {

        // first columns swap
        T tmp = data[tidy * mCols + tidx + (mCols - 1) / 2];
        data[tidy * mCols + tidx + (mCols - 1)/ 2] = data[tidy * mCols + tidx];
        data[tidy * mCols + tidx                 ] = tmp;

        // second columns swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx + (mCols - 1) / 2];
        data[(tidy + mRows / 2) * mCols + tidx + (mCols - 1) / 2] = data[(tidy + mRows / 2) * mCols + tidx];
        data[(tidy + mRows / 2) * mCols + tidx                  ] = tmp;

        // first rows swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx];
        data[(tidy + mRows / 2) * mCols + tidx] = data[tidy * mCols + tidx];
        data[ tidy * mCols + tidx            ] = tmp;

        // second rows swap
        tmp = data[(tidy + mRows / 2) * mCols + tidx + mCols / 2];
        data[(tidy + mRows / 2) * mCols + tidx + mCols / 2] = data[tidy * mCols + tidx + mCols / 2];
        data[ tidy * mCols              + tidx + mCols / 2] = tmp;

        // last column
        if (tidx == 0) {

            tmp = data[(tidy + mRows / 2) * mCols + (mCols - 1)];
            data[(tidy + mRows / 2) * mCols + (mCols - 1)] = data[tidy * mCols + (mCols - 1)];
            data[ tidy              * mCols + (mCols - 1)] = tmp;

            tmp = data[tidy * mCols + tidx + (mCols - 1)];
            data[tidy * mCols + tidx + (mCols - 1)] = data[tidy * mCols + tidx];
            data[tidy * mCols + tidx              ] = tmp;

            tmp = data[(tidy + mRows / 2) * mCols + tidx + (mCols - 1)];
            data[(tidy + mRows / 2) * mCols + tidx + (mCols - 1)] = data[(tidy + mRows / 2) * mCols + tidx];
            data[(tidy + mRows / 2) * mCols + tidx              ] = tmp;
        }
    }
}

template <
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void fftshiftMatrixKernelEvenOddSecondStep(T * CUALGO_RESTRICT data,
                                           unsigned int mRows,
                                           unsigned int mCols)
{

    CUALGO_SHMEM T tile[BlockSizeY][BlockSizeX];

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx < (mCols - 1) / 2 && tidy < mRows) {

        // columns shift
        T last;
        if (tidx == 0)
            last = data[tidy * mCols + tidx];

        if (tidx < BlockSizeX) {
            unsigned int widx = threadIdx.x;
            tile[threadIdx.y][threadIdx.x] = data[tidy * mCols + widx];
            __syncthreads();
            if (widx > 0)
                data[tidy * mCols + widx - 1] = tile[threadIdx.y][threadIdx.x];
            widx += BlockSizeX;
            while ((widx) < ((mCols - 1) / 2)) {
                tile[threadIdx.y][threadIdx.x] = data[tidy * mCols + widx];
                __syncthreads();
                data[tidy * mCols + widx - 1] = tile[threadIdx.y][threadIdx.x];
                widx += BlockSizeX;
            }
            __syncthreads();
            if (tidx == 0)
                data[tidy * mCols + (mCols - 1) / 2 - 1] = last;
        }
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform fftshift on a matrix in place
    * 
    * @details The fftshift operation is performed on both
    *          dimensions
    * 
    * @param[inout] data  pointer to matrix to be shifted
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
    void fftshift2dMatrix(T            *data  ,
                          unsigned int  mRows ,
                          unsigned int  mCols ,
                          cudaStream_t  stream = 0,
                          bool          async = false) {

        if (mRows % 2 == 0 && mCols % 2 == 0) {

            {

                dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
                dim3 blocksPerGrid(div_ceil(mCols / 2, BlockSizeX),
                                   div_ceil(mRows / 2, BlockSizeY));
                print_kernel_config(threadsPerBlock, blocksPerGrid);

                TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                     fftshiftMatrixKernelEvenEven<T>,
                     data, mRows, mCols);
            }
        } else if (mRows % 2 == 0 && mCols % 2 == 1) {

            {

                dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
                dim3 blocksPerGrid(div_ceil((mCols - 1) / 2, BlockSizeX),
                                   div_ceil(mRows / 2, BlockSizeY));
                print_kernel_config(threadsPerBlock, blocksPerGrid);

                TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                     fftshiftMatrixKernelEvenOddFirstStep<T>,
                     data, mRows, mCols);
            }

            {

                dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
                dim3 blocksPerGrid(1, div_ceil(mRows, BlockSizeY));
                print_kernel_config(threadsPerBlock, blocksPerGrid);

                TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                     CUALGO_KERNEL_NAME(fftshiftMatrixKernelEvenOddSecondStep<BlockSizeX, BlockSizeY, T>),
                     data, mRows, mCols);
            }
        }
    }
}
