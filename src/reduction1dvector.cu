/*
 * @file reduction1dvector.cu
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
#include <chrono>

using namespace std::chrono;

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
	if (blockSize >= 8)  sdata[tid] += sdata[tid +  4];
	if (blockSize >= 4)  sdata[tid] += sdata[tid +  2];
	if (blockSize >= 2)  sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void reduce1dKernel(int *g_idata, int *g_odata, unsigned int n) {

	// use dynamic shared memory
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	// load to shared memory
	sdata[tid] = 0;
	while (i < n) {
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1dKernelFlexible(int *g_idata, int *g_odata) {

	// use dynamic shared memory
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce1dVector(int          *g_idata,
                    int          *g_odata,
                    int           size   ,
                    cudaStream_t  stream ,
                    bool          async  ) {

	int threadsPerBlock = size > 1024 ? 1024 : size/2;
	int blocksPerGrid = size / (2*threadsPerBlock) + (size % (2*threadsPerBlock) > 0);
	unsigned int shmem = threadsPerBlock*sizeof(int);

	if (blocksPerGrid == 1) {

		dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
		dim3 threadsPerBlock3(threadsPerBlock, 1, 1);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
		     reduce1dKernelFlexible,
		     g_idata, g_odata);

	} else {

		dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
		dim3 threadsPerBlock3(threadsPerBlock, 1, 1);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, blocksPerGrid*sizeof(int)) );
		switch (threadsPerBlock) {
			case 1024:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<1024>,
			     g_idata, d_buffer, size);
			break;
			case 512:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel< 512>,
			     g_idata, d_buffer, size);
			break;
			case 256:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel< 256>,
			     g_idata, d_buffer, size);
			break;
			case 128:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel< 128>,
			     g_idata, d_buffer, size);
			break;
			case 64:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<  64>,
			     g_idata, d_buffer, size);
			break;
			case 32:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel< 128>,
			     g_idata, d_buffer, size);
			break;
			case 16:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<  16>,
			     g_idata, d_buffer, size);
			break;
			case 8:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<   8>,
			     g_idata, d_buffer, size);
			break;
			case 4:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<   4>,
			     g_idata, d_buffer, size);
			break;
			case 2:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<   2>,
			     g_idata, d_buffer, size);
			break;
			case 1:
			TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
			     reduce1dKernel<   1>,
			     g_idata, d_buffer, size);
			break;
		}

		reduce1dVector(d_buffer, g_odata, blocksPerGrid, stream, async);
		check_cuda( cudaFree ( d_buffer ) );
	}
}
