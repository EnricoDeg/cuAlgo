/*
 * @file convolution1dMatrix.cu
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
#include "internals/gConvolutionCorrelation1dMatrix.hpp"

namespace cuAlgo {

	template<typename T>
	void convolution1dMatrix(T            *R     ,
	                         T            *V     ,
	                         T            *C     ,
	                         unsigned int  N     ,
	                         unsigned int  K     ,
	                         cudaStream_t  stream,
	                         bool          async ) {

		gConvolutionCorrelation1dMatrix<T, convolution_impl>(R, V, C, N, K, stream, async);
	}

	void convolution1dMatrixFloat(float        *R     ,
	                              float        *V     ,
	                              float        *C     ,
	                              unsigned int  N     ,
	                              unsigned int  K     ,
	                              cudaStream_t  stream,
	                              bool          async )
	{

		convolution1dMatrix<float>(R, V, C, N, K, stream, async);
	}

	void convolution1dMatrixDouble(double       *R     ,
	                               double       *V     ,
	                               double       *C     ,
	                               unsigned int  N     ,
	                               unsigned int  K     ,
	                               cudaStream_t  stream,
	                               bool          async )
	{

		convolution1dMatrix<double>(R, V, C, N, K, stream, async);
	}

	void convolution1dMatrixInt(int          *R     ,
	                            int          *V     ,
	                            int          *C     ,
	                            unsigned int  N     ,
	                            unsigned int  K     ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		convolution1dMatrix<int>(R, V, C, N, K, stream, async);
	}

	template void convolution1dMatrix(float  *, float  *, float  *,
	                                  unsigned int, unsigned int,
	                                  cudaStream_t, bool);
	template void convolution1dMatrix(double *, double *, double *,
	                                  unsigned int, unsigned int,
	                                  cudaStream_t, bool);
}
