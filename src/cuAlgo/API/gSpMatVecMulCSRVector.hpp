/*
 * @file gSpMatVecMulCSRVector.hpp
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

template<
unsigned int WarpSize,
typename T>
CUALGO_GLOBAL
void gSpMatVecMulCSRVectorKernel(const unsigned int * CUALGO_RESTRICT columns,
                                 const unsigned int * CUALGO_RESTRICT row_ptr,
                                 const T * CUALGO_RESTRICT values,
                                 const T * CUALGO_RESTRICT x,
                                 T * CUALGO_RESTRICT y,
                                 unsigned int nrows) {

    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id   = thread_id / WarpSize;
    const unsigned int lane      = thread_id % WarpSize;

    const unsigned int row = warp_id;
    T sum = 0;
    if (row < nrows) {
        const unsigned int row_start = row_ptr[row    ];
        const unsigned int row_end   = row_ptr[row + 1];
        for (unsigned int element = row_start + lane; element < row_end; element+=WarpSize) {
            sum += values[element] * x[columns[element]];
        }
    }
    sum = warp_reduce<WarpSize>(sum);
    if (lane == 0 && row < nrows)
        y[row] = sum;
}

namespace cuAlgo {

    /**
    * @brief   Perform sparse matrix-vector multiplication with CSR 
    *          format.
    * 
    * @details The sparse matrix vector multiplication assumes that
    *          the matrix is provided in CSR format.
    *          The vector algorithm provides good performance when the
    *          number of non zero elements is high (more than 64).
    * 
    * @param[in]  columns An integer array of column positions
    *                     where the matrix value is non zero.
    * @param[in]  row_ptr Array of locations in the columns array
    *                     where a new row starts.
    * @param[in] values   An array of non zeros values of the matrix.
    * @param[in]  x       The vector array that multiplies the matrix.
    * @param[out] y       The vector array result of the multiplication.
    * @param[in]  nrows   Number of rows in the matrix.
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template<
    unsigned int BlockSize,
    unsigned int WarpSize,
    typename T>
    void gSpMatVecMulCSRVector(unsigned int *columns,
                               unsigned int *row_ptr,
                               T *values,
                               T *x,
                               T *y,
                               unsigned int nrows,
                               cudaStream_t stream = 0,
                               bool async = false) {

        dim3 threadsPerBlock(BlockSize);
        dim3 blocksPerGrid(div_ceil(nrows, BlockSize / WarpSize));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( threadsPerBlock, blocksPerGrid, 0, stream, async,
              CUALGO_KERNEL_NAME(gSpMatVecMulCSRVectorKernel<WarpSize, T>),
              columns, row_ptr, values, x, y, nrows );
    }
}
