/*
 * @file taper1dmatrix.cu
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
#include "internals/templateShMem.hpp"

template<typename T>
__global__ void taper1dMatrixKernel(      T            *__restrict__ A           ,
                                    const T            *__restrict__ taper       ,
                                    const unsigned int *__restrict__ startIndices,
                                    const unsigned int *__restrict__ endIndices  ,
                                          unsigned int               M           ,
                                          unsigned int               N           ,
                                          unsigned int               taperLength ) {

	// use dynamic shared memory
	// needed for template
	SharedMemory<T> smem;
	T * sdata = smem.getPointer();

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > M || y > N)
		return;

	if (threadIdx.x < taperLength)
		sdata[threadIdx.x] = taper[threadIdx.x];
	__syncthreads();

	if (x < startIndices[y] || x > endIndices[y])
		A[x + y * M] = 0;

	if (x >= startIndices[y] && x < startIndices[y] + taperLength)
		A[x + y * M] *= sdata[x-startIndices[y]];

	if (x <= endIndices[y] && x > endIndices[y] - taperLength)
		A[x + y * M] *= sdata[taperLength - 1 - ( x - ( endIndices[y] - taperLength + 1 ) )];
}

template<typename T>
void taper1dMatrix(T            *A           ,
                   T            *taper       ,
                   unsigned int *startIndices,
                   unsigned int *endIndices  ,
                   unsigned int  M           ,
                   unsigned int  N           ,
                   unsigned int  taperLength ,
                   cudaStream_t  stream      ,
                   bool          async       ) {

	dim3 blocksPerGrid(div_ceil(M, 32), div_ceil(N, 32));
	dim3 threadsPerBlock(32 , 32);
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	unsigned int shmem = taperLength*sizeof(T);

	TIME( blocksPerGrid, threadsPerBlock, shmem, stream, async,
	      taper1dMatrixKernel<T>,
	      A, taper, startIndices, endIndices, M, N, taperLength);
}

namespace cuAlgo {

	void taper1dMatrixInt(int          *A           ,
	                      int          *taper       ,
	                      unsigned int *startIndices,
	                      unsigned int *endIndices  ,
	                      unsigned int  M           ,
	                      unsigned int  N           ,
	                      unsigned int  taperLength ,
	                      cudaStream_t  stream      ,
	                      bool          async       )
	{

		taper1dMatrix<int>(A, taper, startIndices, endIndices, 
		                   M, N, taperLength, stream, async);
	}

	void taper1dMatrixFloat(float        *A           ,
	                        float        *taper       ,
	                        unsigned int *startIndices,
	                        unsigned int *endIndices  ,
	                        unsigned int  M           ,
	                        unsigned int  N           ,
	                        unsigned int  taperLength ,
	                        cudaStream_t  stream      ,
	                        bool          async       )
	{

		taper1dMatrix<float>(A, taper, startIndices, endIndices, 
		                     M, N, taperLength, stream, async);
	}

	void taper1dMatrixDouble(double          *A           ,
	                         double          *taper       ,
	                         unsigned int *startIndices,
	                         unsigned int *endIndices  ,
	                         unsigned int  M           ,
	                         unsigned int  N           ,
	                         unsigned int  taperLength ,
	                         cudaStream_t  stream      ,
	                         bool          async       )
	{

		taper1dMatrix<double>(A, taper, startIndices, endIndices, 
		                      M, N, taperLength, stream, async);
	}
}