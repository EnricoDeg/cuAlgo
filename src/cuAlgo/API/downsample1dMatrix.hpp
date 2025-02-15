/*
 * @file downsample1dMatrix.hpp
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
void downsample1dMatrixDim0(const T * CUALGO_RESTRICT idata,
                            T * CUALGO_RESTRICT odata,
                            unsigned int stride,
                            unsigned int mRows,
                            unsigned int mCols,
                            unsigned int mRowsDown) {

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < mCols && row < mRowsDown) {

        odata[row * mCols + col] = idata[row * stride * mCols + col];
    }
}

template <typename T>
CUALGO_GLOBAL
void downsample1dMatrixDim1(const T * CUALGO_RESTRICT idata,
                            T * CUALGO_RESTRICT odata,
                            unsigned int stride,
                            unsigned int mRows,
                            unsigned int mCols,
                            unsigned int mColsDown) {

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < mColsDown && row < mRows) {

        odata[row * mColsDown + col] = idata[row * mCols + col * stride];
    }
}

namespace cuAlgo {

    /**
    * @brief   downsample operator in 1d on matrix
    * 
    * @details The operation can be applied on both direction
    * 
    * @param[in]  idata pointer to input matrix
    * @param[out] odata pointer to output matrix
    * @param[in]  dim dimension where to apply the dshear operation.
    *             0 for rows and 1 for columns.
    * @param[in]  stride downsample stride on dimension `dim`
    * @param[in]  mRows non-contiguous dimension of the input matrix
    * @param[in]  mCols contiguous dimension of the input matrix
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
    void downsample1dMatrix(T *idata ,
                            T *odata ,
                            unsigned int dim,
                            unsigned int stride,
                            unsigned int mRows,
                            unsigned int mCols,
                            cudaStream_t stream = 0,
                            bool async = false) {

        if (dim == 0) {

            unsigned int mRowsDown = 0;
            for (unsigned int i = 0; i < mRows; i+=stride, ++mRowsDown);

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(mCols, BlockSizeX),
                               div_ceil(mRowsDown, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 downsample1dMatrixDim0<T>,
                 idata, odata, stride, mRows, mCols, mRowsDown);
        } else if (dim == 1) {

            unsigned int mColsDown = 0;
            for (unsigned int i = 0; i < mCols; i+=stride, ++mColsDown);

            dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
            dim3 blocksPerGrid(div_ceil(mColsDown, BlockSizeX),
                               div_ceil(mRows, BlockSizeY));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 downsample1dMatrixDim1<T>,
                 idata, odata, stride, mRows, mCols, mColsDown);
        }
    }
}
