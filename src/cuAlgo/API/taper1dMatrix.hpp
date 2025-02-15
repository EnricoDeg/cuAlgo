/*
 * @file taper1dmatrix.hpp
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
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/templateShMem.hpp"

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

namespace cuAlgo {

/**
 * @brief   Apply 1d taper to matrix.
 * 
 * @details The taper is applied on the fastest 
 *          dimension. Each row of the matrix is associated
 *          with a startIndex and an endIndex. The values
 *          less than startIndex are set to 0. The values
 *          greater than endIndex are also set to 0.
 * 
 * @param[inout]  A            input matrix of size {N,M}
 * @param[in]     taper        taper array of sioze taperLength
 * @param[in]     startIndices start indices to apply taper. 
 *                Values with index less than startIndices are
 *                set to 0. 
 *                The array is of size {N}.
 * @param[in]     endIndices   end indices to apply taper. 
 *                Values with index greater than endIndices are
 *                set to 0. 
 *                The array is of size {N}.
 * @param[in]     M            size of contiguous dimension.
 * @param[in]     N            size of non-contiguous dimension.
 * @param[in]     taperLength  size of taper array.
 * @param[in]  stream CUDA stream where the kernels are launched.
 *                    Default is stream 0 (default stream)
 * @param[in]  async  bool to define if kernels are launched asynchronously
 *                    (without synchronization).
 *                    Default is false (device is synchronized after each kernel launched)
 * 
 * @ingroup algo
 */
	template<typename T>
	void taper1dMatrix(T            *A           ,
	                   T            *taper       ,
	                   unsigned int *startIndices,
	                   unsigned int *endIndices  ,
	                   unsigned int  M           ,
	                   unsigned int  N           ,
	                   unsigned int  taperLength ,
	                   cudaStream_t  stream = 0,
	                   bool          async = false) {

		dim3 blocksPerGrid(div_ceil(M, 32), div_ceil(N, 32));
		dim3 threadsPerBlock(32 , 32);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		unsigned int shmem = taperLength*sizeof(T);

		TIME( blocksPerGrid, threadsPerBlock, shmem, stream, async,
		      taper1dMatrixKernel<T>,
		      A, taper, startIndices, endIndices, M, N, taperLength);
	}
}
