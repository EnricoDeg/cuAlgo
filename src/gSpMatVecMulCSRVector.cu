/*
 * @file gSpMatVecMulCSRVector.cu
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
#include <iostream>
#include "cuAlgo.hpp"
#include "utils.hpp"

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

template<typename T>
__global__ void gSpMatVecMulCSRVectorKernel(const unsigned int * __restrict__ columns,
                                            const unsigned int * __restrict__ row_ptr,
                                            const T            * __restrict__ values ,
                                            const T            * __restrict__ x      ,
                                                  T            * __restrict__ y      ,
                                                  unsigned int                nrows  ) {

	const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int warp_id   = thread_id / WARP_SIZE;
	const unsigned int lane      = thread_id % WARP_SIZE;

	const unsigned int row = warp_id;
	T sum = 0;
	if (row < nrows) {

		const unsigned int row_start = row_ptr[row    ];
		const unsigned int row_end   = row_ptr[row + 1];
		for (unsigned int element = row_start + lane; element < row_end; element+=WARP_SIZE) {
			sum += values[element] * x[columns[element]];
		}
	}
	sum = warp_reduce(sum);
	if (lane == 0 && row < nrows)
		y[row] = sum;
}

template<typename T>
void gSpMatVecMulCSRVector(unsigned int *columns,
                           unsigned int *row_ptr,
                           T            *values ,
                           T            *x      ,
                           T            *y      ,
                           unsigned int  nrows  ,
                           cudaStream_t  stream ,
                           bool          async  ) {

	dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocksPerGrid(div_ceil(nrows, WARPS_PER_BLOCK));
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME( threadsPerBlock, blocksPerGrid, 0, stream, async,
	      gSpMatVecMulCSRVectorKernel<T>,
	      columns, row_ptr, values, x, y, nrows );
}

namespace cuAlgo {

	void gSpMatVecMulCSRVectorInt(unsigned int *columns,
	                              unsigned int *row_ptr,
	                              int          *values ,
	                              int          *x      ,
	                              int          *y      ,
	                              unsigned int  nrows  ,
	                              cudaStream_t  stream ,
	                              bool          async  )
	{

		gSpMatVecMulCSRVector<int>(columns, row_ptr, values , x, y, nrows, stream , async );
	}

	void gSpMatVecMulCSRVectorFloat(unsigned int *columns,
	                                unsigned int *row_ptr,
	                                float        *values ,
	                                float        *x      ,
	                                float        *y      ,
	                                unsigned int  nrows  ,
	                                cudaStream_t  stream ,
	                                bool          async  )
	{

		gSpMatVecMulCSRVector<float>(columns, row_ptr, values , x, y, nrows, stream , async );
	}

	void gSpMatVecMulCSRVectorDouble(unsigned int *columns,
	                                 unsigned int *row_ptr,
	                                 double       *values ,
	                                 double       *x      ,
	                                 double       *y      ,
	                                 unsigned int  nrows  ,
	                                 cudaStream_t  stream ,
	                                 bool          async  )
	{

		gSpMatVecMulCSRVector<double>(columns, row_ptr, values , x, y, nrows, stream , async );
	}
}
