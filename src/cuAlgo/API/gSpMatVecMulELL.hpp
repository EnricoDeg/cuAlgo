/*
 * @file gSpMatVecMulELL.hpp
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
__global__ void gSpMatVecMulELLKernel( const unsigned int * __restrict__ columns         ,
                                       const T            * __restrict__ values          ,
                                       const T            * __restrict__ x               ,
                                             T            * __restrict__ y               ,
                                             unsigned int                nrows           ,
                                             unsigned int                elements_in_rows)
{

	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < nrows) {

		T sum = 0;
		for (unsigned int element = 0; element < elements_in_rows; ++element) {

			const unsigned int offset = row + element * nrows;
			sum += values[offset] * x[columns[offset]];
		}
		y[row] = sum;
	}
}

namespace cuAlgo {

    /**
    * @brief   Perform sparse matrix-vector multiplication with
    *          ELL format.
    * 
    * @details The sparse matrix vector multiplication assumes that
    *          the matrix is provided in ELL format.
    *          The ELL format is similar to the CSR but padding is
    *          used and the matrix is transposed. This means that 
    *          the first non zero elements of all the rows are 
    *          contiguous in memory in the first block.
    *          This format works well when the number of non 
    *          zero elements on each row is similar among all rows.
    *          If only a single row has a much higher number of 
    *          non zero elements, this format will significantly 
    *          increase the memory usage and the performance of
    *          the matrix vector multiplication will drop.
    * 
    * @param[in]  columns          An integer array of column positions
    *                              where the matrix value is non zero.
    * @param[in]  values           An array of non zeros values of the matrix.
    * @param[in]  x                The vector array that multiplies the matrix.
    * @param[out] y                The vector array result of the multiplication.
    * @param[in]  nrows            Number of rows in the matrix.
    * @param[in]  elements_in_rows max number of non zero elements.
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
	template <typename T>
	void gSpMatVecMulELL(unsigned int *columns         ,
	                     T            *values          ,
	                     T            *x               ,
	                     T            *y               ,
	                     unsigned int  nrows           ,
	                     unsigned int  elements_in_rows,
	                     cudaStream_t  stream = 0,
	                     bool          async = false) {

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(nrows, THREADS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME( threadsPerBlock, blocksPerGrid, 0, stream, async,
		      gSpMatVecMulELLKernel<T>,
		      columns, values, x, y, nrows, elements_in_rows );
	}
}