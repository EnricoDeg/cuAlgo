/*
 * @file padarray2dMatrix.cu
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
__global__ void padarrayMatrixKernel(T * __restrict__ idata ,
                                     T * __restrict__ odata ,
                                     unsigned int     nRows ,
                                     unsigned int     nCols ,
                                     unsigned int     mRows ,
                                     unsigned int     mCols )
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

template<typename T>
void padarray2dMatrix(T            *idata ,
                      T            *odata ,
                      unsigned int  nRows ,
                      unsigned int  nCols ,
                      unsigned int  mRows ,
                      unsigned int  mCols ,
                      cudaStream_t  stream,
                      bool          async ) {

	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 blocksPerGrid(div_ceil(nCols, THREADS_PER_BLOCK_X), div_ceil(nRows, THREADS_PER_BLOCK_Y));
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
	     padarrayMatrixKernel<T>,
	     idata, odata, nRows, nCols, mRows, mCols);
}

namespace cuAlgo {

	void padarray2dMatrixFloat(float        *idata ,
	                           float        *odata ,
	                           unsigned int  nRows ,
	                           unsigned int  nCols ,
	                           unsigned int  mRows ,
	                           unsigned int  mCols ,
	                           cudaStream_t  stream,
	                           bool          async )
	{

		padarray2dMatrix<float>(idata, odata, nRows, nCols, mRows, mCols, stream, async);
	}

	void padarray2dMatrixDouble(double       *idata ,
	                            double       *odata ,
	                            unsigned int  nRows ,
	                            unsigned int  nCols ,
	                            unsigned int  mRows ,
	                            unsigned int  mCols ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		padarray2dMatrix<double>(idata, odata, nRows, nCols, mRows, mCols, stream, async);
	}

	void padarray2dMatrixComplexFloat(cuda::std::complex<float> *idata ,
	                                  cuda::std::complex<float> *odata ,
	                                  unsigned int               nRows ,
	                                  unsigned int               nCols ,
	                                  unsigned int               mRows ,
	                                  unsigned int               mCols ,
	                                  cudaStream_t               stream,
	                                  bool                       async ) {

		padarray2dMatrix<cuda::std::complex<float>>(idata, odata, nRows, nCols, mRows, mCols, stream, async);
	}

	void padarray2dMatrixComplexDouble(cuda::std::complex<double> *idata ,
	                                   cuda::std::complex<double> *odata ,
	                                   unsigned int                nRows ,
	                                   unsigned int                nCols ,
	                                   unsigned int                mRows ,
	                                   unsigned int                mCols ,
	                                   cudaStream_t                stream,
	                                   bool                        async ) {

		padarray2dMatrix<cuda::std::complex<double>>(idata, odata, nRows, nCols, mRows, mCols, stream, async);
	}
}
