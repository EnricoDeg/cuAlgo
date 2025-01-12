/*
 * @file dshear1dMatrix.hpp
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
#include "internals/utils.hpp"
#include "config/dshear1dMatrix_config.hpp"

template <typename T>
__global__ void dshear1dMatrixDim0(const T            *__restrict__ idata,
                                         T            *__restrict__ odata,
                                         long int                       k,
                                         unsigned int               mRows,
                                         unsigned int               mCols) {

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    while (col < mCols && row < mRows) {

        long int shift = -k*((long int)mCols / 2 - (long int)col);
        if (abs(shift) > mRows - 1)
            printf("%ld\n", shift);
        if (shift < 0) {

            if (row < mRows+shift)
                odata[row * mCols + col] = idata[(row-shift) * mCols + col];
            else
                odata[row * mCols + col] = idata[(row-(mRows+shift)) * mCols + col];
        } else {

            if (row < shift)
                odata[row * mCols + col] = idata[(mRows-shift+row) * mCols + col];
            else
                odata[row * mCols + col] = idata[(row-shift) * mCols + col];
        }
        col += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void dshear1dMatrixDim1(const T            *__restrict__ idata,
                                         T            *__restrict__ odata,
                                         long int                       k,
                                         unsigned int               mRows,
                                         unsigned int               mCols) {

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    while (col < mCols && row < mRows) {

        long int shift = -k * ((long int)mRows / 2 - (long int)row);
        if (shift < 0) {

            if (col < mCols+shift)
                odata[row * mCols + col] = idata[row * mCols + (col-shift)];
            else
                odata[row * mCols + col] = idata[row * mCols + (col - (mCols + shift))];
        } else {

            if (col < shift)
                odata[row * mCols + col] = idata[row * mCols + (mCols-shift+col)];
            else
                odata[row * mCols + col] = idata[row * mCols + (col-shift)];
        }
        col += blockDim.x * gridDim.x;
    }
}

namespace cuAlgo {

    /**
    * @brief   dshear operator in 1d on matrix
    * 
    * @details The operation can be applied on both direction
    * 
    * @param[in]  idata pointer to input matrix
    * @param[out] odata pointer to output matrix
    * @param[in]  k  long int to define the shift
    * @param[in]  dim dimension where to apply the dshear operation.
    *             0 for rows and 1 for columns.
    * @param[in]  mRows non-contiguous dimension of the input and output matrices
    * @param[in]  mCols contiguous dimension of the input and output matrices
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template<typename T, typename DshearConfig = default_config>
    void dshear1dMatrix(T            *idata ,
                        T            *odata ,
                        long int      k     ,
                        unsigned int  dim   ,
                        unsigned int  mRows ,
                        unsigned int  mCols ,
                        cudaStream_t  stream = 0,
                        bool          async  = false) {

        using config = wrapped_dshear_config<DshearConfig, T>;
        unsigned int target_arch = get_arch();
        const dshear_config_params params = dispatch_target_arch<config>(target_arch);

        const unsigned int block_sizeX      = params.dshear_kernel_config.block_sizeX;
        const unsigned int block_sizeY      = params.dshear_kernel_config.block_sizeY;
        const unsigned int items_per_thread = params.dshear_kernel_config.items_per_thread;

        dim3 threadsPerBlock(block_sizeX, block_sizeY);
        dim3 blocksPerGrid(div_ceil((mCols - 1) / items_per_thread, block_sizeX),
                           div_ceil(mRows, block_sizeY));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        if (dim == 0) {
            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 dshear1dMatrixDim0<T>,
                 idata, odata, k, mRows, mCols);
        } else if (dim == 1) {
            TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
                 dshear1dMatrixDim1<T>,
                 idata, odata, k, mRows, mCols);
        }
    }
}
