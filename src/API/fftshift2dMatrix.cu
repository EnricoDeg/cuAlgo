/*
 * @file fftshift2dMatrix.cu
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
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template <typename T>
__global__ void fftshiftMatrixDim0KernelEven(T * __restrict__  data ,
                                             unsigned int     mRows ,
                                             unsigned int     mCols )
{

	const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx < mCols && tidy < mRows / 2) {

		T tmp = data[(tidy + mRows / 2) * mCols + tidx];
		data[(tidy + mRows / 2) * mCols + tidx] = data[tidy * mCols + tidx];
		data[ tidy * mCols + tidx            ] = tmp;
	}
}

template <typename T>
__global__ void fftshiftMatrixDim1KernelEven(T * __restrict__  data ,
                                             unsigned int     mRows ,
                                             unsigned int     mCols )
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
__global__ void fftshiftMatrixKernelEvenEven(T * __restrict__  data ,
                                             unsigned int     mRows ,
                                             unsigned int     mCols )
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
__global__ void fftshiftMatrixKernelEvenOddFirstStep(T * __restrict__  data ,
                                                     unsigned int     mRows ,
                                                     unsigned int     mCols )
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

template <typename T>
__global__ void fftshiftMatrixKernelEvenOddSecondStep(T * __restrict__  data ,
                                                      unsigned int     mRows ,
                                                      unsigned int     mCols )
{

	__shared__ T tile[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

	const unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx < (mCols - 1) / 2 && tidy < mRows) {

		// columns shift
		T last;
		if (tidx == 0)
			last = data[tidy * mCols + tidx];

		if (tidx < THREADS_PER_BLOCK_X) {
			unsigned int widx = threadIdx.x;
			tile[threadIdx.y][threadIdx.x] = data[tidy * mCols + widx];
			__syncthreads();
			if (widx > 0)
				data[tidy * mCols + widx - 1] = tile[threadIdx.y][threadIdx.x];
			widx += THREADS_PER_BLOCK_X;
			while ((widx) < ((mCols - 1) / 2)) {
				tile[threadIdx.y][threadIdx.x] = data[tidy * mCols + widx];
				__syncthreads();
				data[tidy * mCols + widx - 1] = tile[threadIdx.y][threadIdx.x];
				widx += THREADS_PER_BLOCK_X;
			}
			__syncthreads();
			if (tidx == 0)
				data[tidy * mCols + (mCols - 1) / 2 - 1] = last;
		}
	}
}

template<typename T>
void fftshift2dMatrix(T            *data  ,
                      unsigned int  mRows ,
                      unsigned int  mCols ,
                      cudaStream_t  stream,
                      bool          async ) {

	if (mRows % 2 == 0 && mCols % 2 == 0) {

		{
			dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
			dim3 blocksPerGrid(div_ceil(mCols / 2, THREADS_PER_BLOCK_X), div_ceil(mRows / 2, THREADS_PER_BLOCK_Y));
			print_kernel_config(threadsPerBlock, blocksPerGrid);

			TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
			     fftshiftMatrixKernelEvenEven<T>,
			     data, mRows, mCols);
		}
	} else if (mRows % 2 == 0 && mCols % 2 == 1) {

		{
			dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
			dim3 blocksPerGrid(div_ceil((mCols - 1) / 2, THREADS_PER_BLOCK_X), div_ceil(mRows / 2, THREADS_PER_BLOCK_Y));
			print_kernel_config(threadsPerBlock, blocksPerGrid);

			TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
			     fftshiftMatrixKernelEvenOddFirstStep<T>,
			     data, mRows, mCols);
		}

		{
			dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
			dim3 blocksPerGrid(1, div_ceil(mRows, THREADS_PER_BLOCK_Y));
			print_kernel_config(threadsPerBlock, blocksPerGrid);

			TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
			     fftshiftMatrixKernelEvenOddSecondStep<T>,
			     data, mRows, mCols);
		}
	}
}

namespace cuAlgo {

	void fftshift2dMatrixFloat(float        *data  ,
	                           unsigned int  mRows ,
	                           unsigned int  mCols ,
	                           cudaStream_t  stream,
	                           bool          async )
	{

		fftshift2dMatrix<float>(data, mRows, mCols, stream, async);
	}

	void fftshift2dMatrixDouble(double       *data  ,
	                            unsigned int  mRows ,
	                            unsigned int  mCols ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		fftshift2dMatrix<double>(data, mRows, mCols, stream, async);
	}
}
