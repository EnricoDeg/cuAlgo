/*
 * @file normalizeVector.cu
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
#include "cuAlgo.hpp"
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template<typename T>
__global__ void normalizeKernel(T            * __restrict__ data,
                                T            * __restrict__ norm,
                                unsigned int                size) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < size) {

		data[i] /= (*norm);
		i += gridDim.x * blockDim.x;
	}
}

namespace cuAlgo {

	template<typename T>
	void normalizeVector(T            *g_idata,
	                     unsigned int  size   ,
	                     cudaStream_t  stream ,
	                     bool          async  ) {

		T * g_odata;
		check_cuda( cudaMalloc(&g_odata, sizeof(T)) );

		normL1Vector<T>(g_idata, g_odata, size, stream, async);

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(size, THREADS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     normalizeKernel<T>,
		     g_idata, g_odata, size);

		check_cuda( cudaFree ( g_odata ) );
	}

	void normalizeVectorFloat(float        *g_idata,
	                          unsigned int  size   ,
	                          cudaStream_t  stream ,
	                          bool          async  )
	{

		normalizeVector<float>(g_idata, size, stream, async);
	}

	void normalizeVectorDouble(double       *g_idata,
	                           unsigned int  size   ,
	                           cudaStream_t  stream ,
	                           bool          async  )
	{

		normalizeVector<double>(g_idata, size, stream, async);
	}

	void normalizeVectorInt(int          *g_idata,
	                        unsigned int  size   ,
	                        cudaStream_t  stream ,
	                        bool          async  )
	{

		normalizeVector<int>(g_idata, size, stream, async);
	}

	template void normalizeVector(float  *,
	                              unsigned int,
	                              cudaStream_t, bool);
	template void normalizeVector(double *,
	                              unsigned int,
	                              cudaStream_t, bool);
}
