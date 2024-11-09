/*
 * @file fftshift1dvector.cu
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

#define TILE_DIM 1024

template <typename T>
__global__ void fftshiftVectorKernelEven(T * __restrict__ idata ,
                                         T * __restrict__ odata ,
                                         unsigned int     size  )
{

	const unsigned int tid_in = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid_in < size) {

		const unsigned int tid_out = tid_in < size / 2 ?
		                             tid_in + size / 2 :
		                             tid_in - size / 2 ;
		odata[tid_out] = idata[tid_in];
	}
}

template <typename T>
__global__ void fftshiftVectorKernelOdd(T * __restrict__ idata ,
                                        T * __restrict__ odata ,
                                        unsigned int     size  )
{

	const unsigned int tid_in = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid_in < size) {

		const unsigned int tid_out = tid_in <= ( size - 1 ) / 2 ?
		                             tid_in +  ( size - 1 ) / 2 :
		                             tid_in -  ( size + 1 ) / 2 ;
		odata[tid_out] = idata[tid_in];
	}
}

template <typename T>
void fftshiftVector(T            *idata ,
                    T            *odata ,
                    unsigned int  size,
                    cudaStream_t  stream,
                    bool          async ) {

	dim3 blocksPerGrid3(size / TILE_DIM, 1, 1);
	dim3 threadsPerBlock3(TILE_DIM, 1, 1);

	print_kernel_config(threadsPerBlock3, blocksPerGrid3) ;

	if (size % 2 == 0) {

		TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
		     fftshiftVectorKernelEven<T>,
		     idata, odata, size);
	} else {

		TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
		     fftshiftVectorKernelOdd<T>,
		     idata, odata, size);
	}
}

namespace cuAlgo {

	void fftshiftVectorFloat(float        *idata ,
	                         float        *odata ,
	                         unsigned int  size  ,
	                         cudaStream_t  stream,
	                         bool          async ) {

		fftshiftVector<float>(idata, odata, size, stream, async);
	}

	void fftshiftVectorDouble(double       *idata ,
	                          double       *odata ,
	                          unsigned int  size  ,
	                          cudaStream_t  stream,
	                          bool          async ) {

		fftshiftVector<double>(idata, odata, size, stream, async);
	}

	void fftshiftVectorInt(int          *idata ,
	                       int          *odata ,
	                       unsigned int  size  ,
	                       cudaStream_t  stream,
	                       bool          async ) {

		fftshiftVector<int>(idata, odata, size, stream, async);
	}
}