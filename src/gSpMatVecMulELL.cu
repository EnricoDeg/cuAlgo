/*
 * @file gSpMatVecMulELL.cu
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
#include "cuAlgo.hpp"
#include "utils.hpp"

#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

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

template <typename T>
void gSpMatVecMulELL(unsigned int *columns         ,
                     T            *values          ,
                     T            *x               ,
                     T            *y               ,
                     unsigned int  nrows           ,
                     unsigned int  elements_in_rows,
                     cudaStream_t  stream          ,
                     bool          async           ) {

	dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocksPerGrid(div_ceil(nrows, THREADS_PER_BLOCK));
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME( threadsPerBlock, blocksPerGrid, 0, stream, async,
	      gSpMatVecMulELLKernel<T>,
	      columns, values, x, y, nrows, elements_in_rows );

}

namespace cuAlgo {

	void gSpMatVecMulELLInt(unsigned int *columns         ,
	                        int          *values          ,
	                        int          *x               ,
	                        int          *y               ,
	                        unsigned int  nrows           ,
	                        unsigned int  elements_in_rows,
	                        cudaStream_t  stream          ,
	                        bool          async           )
	{

		gSpMatVecMulELL<int>(columns, values, x, y, nrows, elements_in_rows, stream, async);
	}

	void gSpMatVecMulELLFloat(unsigned int *columns         ,
	                          float        *values          ,
	                          float        *x               ,
	                          float        *y               ,
	                          unsigned int  nrows           ,
	                          unsigned int  elements_in_rows,
	                          cudaStream_t  stream          ,
	                          bool          async           )
	{

		gSpMatVecMulELL<float>(columns, values, x, y, nrows, elements_in_rows, stream, async);
	}

	void gSpMatVecMulELLDouble(unsigned int *columns         ,
	                           double       *values          ,
	                           double       *x               ,
	                           double       *y               ,
	                           unsigned int  nrows           ,
	                           unsigned int  elements_in_rows,
	                           cudaStream_t  stream          ,
	                           bool          async           )
	{

		gSpMatVecMulELL<double>(columns, values, x, y, nrows, elements_in_rows, stream, async);
	}
}