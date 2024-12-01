/*
 * @file convolution2dMatrixDD.cu
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
#include "cuAlgo.h"
#include "internals/templateShMem.hpp"
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template <typename T>
__global__ void convolution2dMatrixDDKernel(      T *__restrict__ odata ,
                                            const T *__restrict__ idata ,
                                            const T *__restrict__ filter,
                                            unsigned int          mRows ,
                                            unsigned int          mCols ,
                                            unsigned int          fRows ,
                                            unsigned int          fCols ) {

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
		if (row == 0 && col == 0)
			printf("%d\n", low_mr);
		T tmp = 0;
#pragma unroll
		for (unsigned int mr = low_mr; mr <= high_mr; ++mr)
			for (unsigned int mc = low_mc; mc <= high_mc; ++mc) {
				// if (row == 0 && col == 0)
				// 	printf( "%f, %f\n", filter[(row - mr) * fCols + (col - mc)] , idata[mr * mCols + mc] );
				tmp += idata[mr * mCols + mc] * filter[(row - mr) * fCols + (col - mc)];
			}
		odata[row*totalCols+col] = tmp;
	}
}

namespace cuAlgo {

	template<typename T>
	void convolution2dMatrixDD(T            * odata ,
	                           T            * idata ,
	                           T            * filter,
	                           unsigned int   mRows ,
	                           unsigned int   mCols ,
	                           unsigned int   fRows ,
	                           unsigned int   fCols ,
	                           cudaStream_t   stream,
	                           bool           async ) {

		unsigned int totalRows = mRows + fRows - 1;
		unsigned int totalCols = mCols + fCols - 1;
		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(totalCols, THREADS_PER_BLOCK_X), div_ceil(totalRows, THREADS_PER_BLOCK_Y));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		unsigned int shmem = fRows*fCols*sizeof(T);

		// compute convolution using padded data
		TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async,
		     convolution2dMatrixDDKernel<T>,
		     odata, idata, filter, mRows, mCols, fRows, fCols);
	}

	void convolution2dMatrixDDFloat(float        * odata ,
	                                float        * idata ,
	                                float        * filter,
	                                unsigned int   mRows ,
	                                unsigned int   mCols ,
	                                unsigned int   fRows ,
	                                unsigned int   fCols ,
	                                cudaStream_t   stream,
	                                bool           async )
	{

		convolution2dMatrixDD<float>(odata, idata, filter, mRows, mCols, fRows, fCols, stream, async);
	}

	void convolution2dMatrixDDDouble(double       * odata ,
	                                 double       * idata ,
	                                 double       * filter,
	                                 unsigned int   mRows ,
	                                 unsigned int   mCols ,
	                                 unsigned int   fRows ,
	                                 unsigned int   fCols ,
	                                 cudaStream_t   stream,
	                                 bool           async )
	{

		convolution2dMatrixDD<double>(odata, idata, filter, mRows, mCols, fRows, fCols, stream, async);
	}

	template void convolution2dMatrixDD(float  *, float  *, float  *,
	                                    unsigned int, unsigned int,
	                                    unsigned int, unsigned int,
	                                    cudaStream_t, bool);
	template void convolution2dMatrixDD(double *, double *, double *,
	                                    unsigned int, unsigned int,
	                                    unsigned int, unsigned int,
	                                    cudaStream_t, bool);
}
