/*
 * @file padarray2dMatrix.hpp
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
void padarrayMatrixKernel(T * CUALGO_RESTRICT idata,
                          T * CUALGO_RESTRICT odata,
                          unsigned int nRows,
                          unsigned int nCols,
                          unsigned int mRows,
                          unsigned int mCols)
{

    const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx < nCols && tidy < nRows) {

        unsigned int offsetRows = ( nRows - mRows ) / 2 + ( nRows - mRows ) % 2;
        unsigned int offsetCols = ( nCols - mCols ) / 2 + ( nCols - mCols ) % 2;

        if (tidx >= offsetCols && tidx < offsetCols + mCols &&
            tidy >= offsetRows && tidy < offsetRows + mRows  ) {
            odata[tidy * nCols + tidx] = idata[(tidy-offsetRows) * mCols + (tidx - offsetCols)];
        } else {
            odata[tidy * nCols + tidx] = 0;
        }
    }
}

namespace cuAlgo {

    /**
    * @brief   Pad matrix
    * 
    * @param[in]  idata pointer to input matrix
    * @param[out] odata pointer to output matrix
    * @param[in]  nRows non-contiguous dimension of odata
    * @param[in]  nCols contiguous dimension of odata
    * @param[in]  mRows non-contiguous dimension of idata
    * @param[in]  mCols contiguous dimension of idata
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
    void padarray2dMatrix(T *idata,
                          T *odata,
                          unsigned int nRows,
                          unsigned int nCols,
                          unsigned int mRows,
                          unsigned int mCols,
                          cudaStream_t stream = 0,
                          bool async = false) {

        dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
        dim3 blocksPerGrid(div_ceil(nCols, BlockSizeX),
                           div_ceil(nRows, BlockSizeY));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
             padarrayMatrixKernel<T>,
             idata, odata, nRows, nCols, mRows, mCols);
    }
}
