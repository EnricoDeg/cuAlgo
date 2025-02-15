/*
 * @file convolution2dMatrixDD.hpp
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
#include "cuAlgo/internals/templateShMem.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/kernelParameters.hpp"

template <typename T>
CUALGO_GLOBAL
void convolution2dMatrixDDKernel(T * CUALGO_RESTRICT odata,
                                 const T * CUALGO_RESTRICT idata,
                                 const T * CUALGO_RESTRICT filter,
                                 unsigned int mRows,
                                 unsigned int mCols,
                                 unsigned int fRows,
                                 unsigned int fCols) {

    // use dynamic shared memory
    // neeeded for template
    SharedMemory<T> smem;
    T * sdata = smem.getPointer();

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int totalRows = mRows + fRows - 1;
    unsigned int totalCols = mCols + fCols - 1;

    if (row < totalRows && col < totalCols ) {

        // load shared memory with filter
        unsigned int sidx = threadIdx.x;
        unsigned int sidy = threadIdx.y;
        while (sidx < fCols && sidy < fRows) {
            sdata[sidy * fCols + sidx] = filter[sidy * fCols + sidx];
            sidx += blockDim.x;
            sidy += blockDim.y;
        }
        __syncthreads();

        // compute ranges
        unsigned int low_mr  = (int)row - (int)fRows + 1 < 0 ? 0         : row - fRows + 1;
        unsigned int high_mr = row > mRows - 1          ? mRows - 1 : row            ;
        unsigned int low_mc  = (int)col - (int)fCols + 1 < 0 ? 0         : col - fCols + 1;
        unsigned int high_mc = col > mCols - 1          ? mCols - 1 : col            ;
        T tmp = 0;
        CUALGO_UNROLL
        for (unsigned int mr = low_mr; mr <= high_mr; ++mr)
            for (unsigned int mc = low_mc; mc <= high_mc; ++mc) {
                tmp += idata[mr * mCols + mc] * filter[(row - mr) * fCols + (col - mc)];
            }
        odata[row*totalCols+col] = tmp;
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform 2D convolution in data domain on the input matrices.
    * 
    * @details The convolution is done in the data domain without Fourier
    *          Transform.
    * 
    * @param[out] odata  pointer to the output matrix for the convolution.
    *                    The matrix has dimensions {mRows+fRows-1, mCols+fCols-1}.
    * @param[in]  idata  pointer to the input matrix for the convolution.
    *                    The matrix has dimensions {mRows, mCols}.
    * @param[in]  filter pointer to the kernel of the convolution.
    *                    The matrix has dimension {fRows, fCols}.
    * @param[in]  mRows  non-contiguous dimension of the input matrix
    * @param[in]  mCols  contiguous dimension of the input matrix
    * @param[in]  fRows  non-contiguous dimensions of the kernel matrix
    * @param[in]  fCols  contiguous dimension of the kernel matrix
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
    typename T
    >
    void convolution2dMatrixDD(T * odata,
                               T * idata,
                               T * filter,
                               unsigned int mRows,
                               unsigned int mCols,
                               unsigned int fRows,
                               unsigned int fCols,
                               cudaStream_t stream = 0,
                               bool async = false) {

        unsigned int totalRows = mRows + fRows - 1;
        unsigned int totalCols = mCols + fCols - 1;
        dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
        dim3 blocksPerGrid(div_ceil(totalCols, BlockSizeX),
                           div_ceil(totalRows, BlockSizeY));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        unsigned int shmem = fRows*fCols*sizeof(T);

        // compute convolution using padded data
        TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async,
             convolution2dMatrixDDKernel<T>,
             odata, idata, filter, mRows, mCols, fRows, fCols);
    }
}
