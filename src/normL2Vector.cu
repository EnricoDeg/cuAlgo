/*
 * @file normL2Vector.cu
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
#include "cuAlgoInternal.hpp"
#include "gReduction1dVector.hpp"

template<typename T>
void normL2Vector(T            *g_idata,
                  T            *g_odata,
                  unsigned int  size   ,
                  cudaStream_t  stream ,
                  bool          async  ) {

	unsigned int threadsPerBlock = size > 1024 ? 1024 : size / 2;
	unsigned int blocksPerGrid = size / (2*threadsPerBlock) + (size % (2*threadsPerBlock) > 0);

	if (blocksPerGrid == 1) {

		gReduction1dVectorFlexible<T, normL2_impl>(g_idata        ,
		                                           g_odata        ,
		                                           size           ,
		                                           stream         ,
		                                           async          ,
		                                           threadsPerBlock);

	} else {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, blocksPerGrid*sizeof(T)) );

		gReduction1dVectorPower2<T, normL2_impl>(g_idata,
		                                         d_buffer,
		                                         size   ,
		                                         stream ,
		                                         async  ,
		                                         threadsPerBlock,
		                                         blocksPerGrid) ;

		reduction1dVector<T>(d_buffer, g_odata, blocksPerGrid, stream, async);
		check_cuda( cudaFree ( d_buffer ) );
	}
}

namespace cuAlgo {

	void normL2VectorFloat(float        *g_idata,
	                       float        *g_odata,
	                       unsigned int  size   ,
	                       cudaStream_t  stream ,
	                       bool          async  )
	{

		normL2Vector<float>(g_idata, g_odata, size, stream, async);
	}

	void normL2VectorDouble(double       *g_idata,
	                        double       *g_odata,
	                        unsigned int  size   ,
	                        cudaStream_t  stream ,
	                        bool          async  )
	{

		normL2Vector<double>(g_idata, g_odata, size, stream, async);
	}

	void normL2VectorInt(int          *g_idata,
	                     int          *g_odata,
	                     unsigned int  size   ,
	                     cudaStream_t  stream ,
	                     bool          async  )
	{

		normL2Vector<int>(g_idata, g_odata, size, stream, async);
	}
}
