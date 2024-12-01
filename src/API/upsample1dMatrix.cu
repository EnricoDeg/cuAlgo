/*
 * @file upsample1dMatrix.cu
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
__global__ void upsample1dMatrixDim0(const T            *__restrict__ idata   ,
                                           T            *__restrict__ odata   ,
                                           unsigned int               nzeros  ,
                                           unsigned int               mRows   ,
                                           unsigned int               mCols   ,
                                           unsigned int               mRowsUp ) {

	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < mCols && row < mRowsUp) {

		if (row % (nzeros+1) == 0) {
			odata[row * mCols + col] = idata[row / (nzeros+1) * mCols + col];
		} else {
			odata[row * mCols + col] = 0;
		}
	}
}

template <typename T>
__global__ void upsample1dMatrixDim1(const T            *__restrict__ idata   ,
                                           T            *__restrict__ odata   ,
                                           unsigned int               nzeros  ,
                                           unsigned int               mRows   ,
                                           unsigned int               mCols   ,
                                           unsigned int               mColsUp ) {

	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < mColsUp && row < mRows) {

		if (col % (nzeros+1) == 0) {
			odata[row * mColsUp + col] = idata[row * mCols + col / (nzeros+1)];
		} else {
			odata[row * mColsUp + col] = 0;
		}
	}
}

template<typename T>
void upsample1dMatrix(T            *idata ,
                      T            *odata ,
                      unsigned int  dim   ,
                      unsigned int  nzeros,
                      unsigned int  mRows ,
                      unsigned int  mCols ,
                      cudaStream_t  stream,
                      bool          async ) {

	if (dim == 0) {

		unsigned int mRowsUp = (mRows-1)*(nzeros)+mRows;

		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(mCols, THREADS_PER_BLOCK_X), div_ceil(mRowsUp, THREADS_PER_BLOCK_Y));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     upsample1dMatrixDim0<T>,
		     idata, odata, nzeros, mRows, mCols, mRowsUp);
	} else if (dim == 1) {

		unsigned int mColsUp = (mCols-1)*(nzeros)+mCols;

		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(mColsUp, THREADS_PER_BLOCK_X), div_ceil(mRows, THREADS_PER_BLOCK_Y));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     upsample1dMatrixDim1<T>,
		     idata, odata, nzeros, mRows, mCols, mColsUp);
	}
}

namespace cuAlgo {

	void upsample1dMatrixFloat(float        *idata ,
	                           float        *odata ,
	                           unsigned int  dim   ,
	                           unsigned int  nzeros,
	                           unsigned int  mRows ,
	                           unsigned int  mCols ,
	                           cudaStream_t  stream,
	                           bool          async )
	{

		upsample1dMatrix<float>(idata, odata, dim, nzeros, mRows, mCols, stream, async);
	}

	void upsample1dMatrixDouble(double       *idata ,
	                            double       *odata ,
	                            unsigned int  dim   ,
	                            unsigned int  nzeros,
	                            unsigned int  mRows ,
	                            unsigned int  mCols ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		upsample1dMatrix<double>(idata, odata, dim, nzeros, mRows, mCols, stream, async);
	}
}
