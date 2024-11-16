/*
 * @file gOperationAndReduction1dVector.hpp
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

#ifndef GOPERATIONANDREDUCTION1DVECTOR_H
#define GOPERATIONANDREDUCTION1DVECTOR_H

#include <cuda.h>
#include "internals/operations.hpp"
#include "internals/templateShMem.hpp"
#include "internals/utils.hpp"
#include "internals/reductionShMem.hpp"

template <unsigned int blockSize, typename T, template<typename> class op_t>
__global__ void operationAndReduction1dKernel(T *g_idata1, T *g_idata2, T *g_odata, unsigned int n, op_t<T> Op) {

	// use dynamic shared memory
	// needed for template
	SharedMemory<T> smem;
	T * sdata = smem.getPointer();

	// parameters
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	// load multiple elements to shared memory
	sdata[tid] = 0;
	while (i < n) {
		T a = Op.globalMemory( &g_idata1[i], &g_idata2[i], &g_idata1[i+blockSize], &g_idata2[i+blockSize] );
		Op.loadSharedMemory(&sdata[tid], &a);
		i += gridSize;
	}
	__syncthreads();

	// do reduction in shared mem
	blockReduceShMemUnroll<blockSize, T, op_t>(sdata, tid, Op);

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<typename T, template<typename> class op_t>
__global__ void operationAndReduction1dKernelFlexible(T *g_idata1, T *g_idata2, T *g_odata, op_t<T> Op) {

	// use dynamic shared memory
	// neeeded for template
	SharedMemory<T> smem;
	T * sdata = smem.getPointer();

	// parameters
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	// load one element to shared mem
	sdata[tid] = Op.globalMemory( &g_idata1[i], &g_idata2[i], &g_idata1[i+blockDim.x], &g_idata2[i+blockDim.x] );
	__syncthreads();

	// do reduction in shared mem
	blockReduceShMem<T, op_t>(sdata, tid, Op);

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<typename T, template<typename> class op_t>
void gOperationAndReduction1dVectorFlexible(T            *g_idata1       ,
                                            T            *g_idata2       ,
                                            T            *g_odata        ,
                                            unsigned int  size           ,
                                            cudaStream_t  stream         ,
                                            bool          async          ,
                                            unsigned int  threadsPerBlock) {

	unsigned int shmem = threadsPerBlock*sizeof(T);
	op_t<T> op;

	dim3 blocksPerGrid3(1, 1, 1);
	dim3 threadsPerBlock3(threadsPerBlock, 1, 1);
	print_kernel_config(threadsPerBlock3, blocksPerGrid3);

	TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
	     operationAndReduction1dKernelFlexible<T COMMA op_t>,
	     g_idata1, g_idata2, g_odata, op);
}

template<typename T, template<typename> class op_t>
void gOperationAndReduction1dVectorPower2(T            *g_idata1       ,
                                          T            *g_idata2       ,
                                          T            *d_buffer       ,
                                          unsigned int  size           ,
                                          cudaStream_t  stream         ,
                                          bool          async          ,
                                          unsigned int  threadsPerBlock,
                                          unsigned int  blocksPerGrid  ) {

	unsigned int shmem = threadsPerBlock*sizeof(T);
	op_t<T> op;
	dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
	dim3 threadsPerBlock3(threadsPerBlock, 1, 1);
	print_kernel_config(threadsPerBlock3, blocksPerGrid3);

	switch (threadsPerBlock) {
		case 1024:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<1024 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 512:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel< 512 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 256:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel< 256 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 128:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel< 128 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 64:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<  64 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 32:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel< 32 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 16:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<  16 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 8:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<   8 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 4:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<   4 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 2:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<   2 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
		case 1:
		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     operationAndReduction1dKernel<   1 COMMA T COMMA op_t>,
		     g_idata1, g_idata2, d_buffer, size, op);
		break;
	}
}

#endif