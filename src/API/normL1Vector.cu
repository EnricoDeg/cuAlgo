/*
 * @file normL1Vector.cu
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
#include "internals/cuAlgoInternal.hpp"
#include "internals/gReduction1dVector.hpp"

template<typename T>
void normL1Vector(T            *g_idata,
                  T            *g_odata,
                  unsigned int  size   ,
                  cudaStream_t  stream ,
                  bool          async  ) {

	unsigned int threadsPerBlock = size > 1024 ? 1024 : size / 2;
	unsigned int blocksPerGrid = size / (2*threadsPerBlock) + (size % (2*threadsPerBlock) > 0);

	if (blocksPerGrid == 1) {

		gReduction1dVectorFlexible<T, normL1_impl>(g_idata        ,
		                                           g_odata        ,
		                                           size           ,
		                                           stream         ,
		                                           async          ,
		                                           threadsPerBlock);

	} else {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, blocksPerGrid*sizeof(T)) );

		gReduction1dVectorPower2<T, normL1_impl>(g_idata,
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

	void normL1VectorFloat(float        *g_idata,
	                       float        *g_odata,
	                       unsigned int  size   ,
	                       cudaStream_t  stream ,
	                       bool          async  )
	{

		normL1Vector<float>(g_idata, g_odata, size, stream, async);
	}

	void normL1VectorDouble(double       *g_idata,
	                        double       *g_odata,
	                        unsigned int  size   ,
	                        cudaStream_t  stream ,
	                        bool          async  )
	{

		normL1Vector<double>(g_idata, g_odata, size, stream, async);
	}

	void normL1VectorInt(int          *g_idata,
	                     int          *g_odata,
	                     unsigned int  size   ,
	                     cudaStream_t  stream ,
	                     bool          async  )
	{

		normL1Vector<int>(g_idata, g_odata, size, stream, async);
	}
}

template void  normL1Vector( float *, float *, unsigned int , cudaStream_t , bool );
template void  normL1Vector( double *, double *, unsigned int , cudaStream_t , bool );
template void  normL1Vector( int *, int *, unsigned int , cudaStream_t , bool );

