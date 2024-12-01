/*
 * @file dotProduct1dVector.cu
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
#include "internals/gOperationAndReduction1dVector.hpp"

namespace cuAlgo {

	template<typename T>
	void dotProduct1dVector(T            *g_idata1,
	                        T            *g_idata2,
	                        T            *g_odata ,
	                        unsigned int  size    ,
	                        cudaStream_t  stream  ,
	                        bool          async   ) {

		unsigned int threadsPerBlock = size > 1024 ? 1024 : size / 2;
		unsigned int blocksPerGrid = size / (2*threadsPerBlock) + (size % (2*threadsPerBlock) > 0);

		if (blocksPerGrid == 1) {

			gOperationAndReduction1dVectorFlexible<T, dotProduct_impl>(g_idata1       ,
			                                                           g_idata2       ,
			                                                           g_odata        ,
			                                                           size           ,
			                                                           stream         ,
			                                                           async          ,
			                                                           threadsPerBlock);
		} else {

			T * d_buffer;
			check_cuda( cudaMalloc(&d_buffer, blocksPerGrid*sizeof(T)) );

			gOperationAndReduction1dVectorPower2<T, dotProduct_impl>(g_idata1       ,
			                                                         g_idata2       ,
			                                                         d_buffer       ,
			                                                         size           ,
			                                                         stream         ,
			                                                         async          ,
			                                                         threadsPerBlock,
			                                                         blocksPerGrid  ) ;

			reduction1dVector<T>(d_buffer, g_odata, blocksPerGrid, stream, async);

			check_cuda( cudaFree ( d_buffer ) );
		}
	}

	void dotProduct1dVectorFloat(float        *g_idata1,
	                             float        *g_idata2,
	                             float        *g_odata ,
	                             unsigned int  size    ,
	                             cudaStream_t  stream  ,
	                             bool          async   )
	{

		dotProduct1dVector<float>(g_idata1, g_idata2, g_odata, size, stream, async);
	}

	void dotProcuct1dVectorDouble(double       *g_idata1,
	                              double       *g_idata2,
	                              double       *g_odata ,
	                              unsigned int  size    ,
	                              cudaStream_t  stream  ,
	                              bool          async   )
	{

		dotProduct1dVector<double>(g_idata1, g_idata2, g_odata, size, stream, async);
	}

	void dotProduct1dVectorInt(int          *g_idata1,
	                           int          *g_idata2,
	                           int          *g_odata ,
	                           unsigned int  size    ,
	                           cudaStream_t  stream  ,
	                           bool          async   )
	{

		dotProduct1dVector<int>(g_idata1, g_idata2, g_odata, size, stream, async);
	}

	template void dotProduct1dVector(float  *, float  *, float  *,
	                                 unsigned int,
	                                 cudaStream_t, bool);
	template void dotProduct1dVector(double *, double *, double *,
	                                 unsigned int,
	                                 cudaStream_t, bool);
}
