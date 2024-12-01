/*
 * @file dshear1dMatrix.cu
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
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template <typename T>
__global__ void dshear1dMatrixDim0(const T            *__restrict__ idata,
                                         T            *__restrict__ odata,
                                         long int                       k,
                                         unsigned int               mRows,
                                         unsigned int               mCols) {

	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < mCols && row < mRows) {

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

	}
}

template <typename T>
__global__ void dshear1dMatrixDim1(const T            *__restrict__ idata,
                                         T            *__restrict__ odata,
                                         long int                       k,
                                         unsigned int               mRows,
                                         unsigned int               mCols) {

	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < mCols && row < mRows) {

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
	}
}

namespace cuAlgo {

	template<typename T>
	void dshear1dMatrix(T            *idata ,
	                    T            *odata ,
	                    long int      k     ,
	                    unsigned int  dim   ,
	                    unsigned int  mRows ,
	                    unsigned int  mCols ,
	                    cudaStream_t  stream,
	                    bool          async ) {

		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(mCols, THREADS_PER_BLOCK_X), div_ceil(mRows, THREADS_PER_BLOCK_Y));
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

	void dshear1dMatrixFloat(float        *idata ,
	                         float        *odata ,
	                         long int      k     ,
	                         unsigned int  dim   ,
	                         unsigned int  mRows ,
	                         unsigned int  mCols ,
	                         cudaStream_t  stream,
	                         bool          async )
	{

		dshear1dMatrix<float>(idata, odata, k, dim, mRows, mCols, stream, async);
	}

	void dshear1dMatrixDouble(double       *idata ,
	                          double       *odata ,
	                          long int      k     ,
	                          unsigned int  dim   ,
	                          unsigned int  mRows ,
	                          unsigned int  mCols ,
	                          cudaStream_t  stream,
	                          bool          async )
	{

		dshear1dMatrix<double>(idata, odata, k, dim, mRows, mCols, stream, async);
	}

	template void dshear1dMatrix(float  *, float  *,
	                             long int,
	                             unsigned int, unsigned int, unsigned int,
	                             cudaStream_t, bool);
	template void dshear1dMatrix(double *, double *,
	                             long int,
	                             unsigned int, unsigned int, unsigned int,
	                             cudaStream_t, bool);
}
