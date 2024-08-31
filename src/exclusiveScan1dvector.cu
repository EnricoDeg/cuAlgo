/*
 * @file exclusiveScan1dvector.cu
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
#include "utils.hpp"
#include "templateShMem.hpp"

#define THREADS_PER_BLOCK 512
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

__global__ void exclusiveScan1dKernelBlock(const int          *__restrict__ g_idata,
                                                 int          *__restrict__ g_odata,
                                                 unsigned int               size   ) {

	extern __shared__ int temp[];
	unsigned int thid = threadIdx.x;
	unsigned int offset = 1;
	temp[2*thid  ] = g_idata[2*thid  ];
	temp[2*thid+1] = g_idata[2*thid+1];

	for (unsigned int d = size>>1 ; d > 0 ; d >>=1) {

		__syncthreads();
		if (thid < d) {

			unsigned int ai = offset * (2 * thid + 1) - 1;
			unsigned int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0)
		temp[ size - 1 ] = 0;

	for (unsigned int d = 1; d < size; d *= 2) {

		offset >>= 1;
		__syncthreads();
		if (thid < d) {

			unsigned int ai = offset * (2 * thid + 1) - 1;
			unsigned int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2*thid] = temp[2*thid];
	g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void exclusiveScan1dKernelMultiBlock(const int          *__restrict__ g_idata,
                                                      int          *__restrict__ g_odata,
                                                      int          *__restrict__ sums   ,
                                                      unsigned int               size   ) {

	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * size;
	
	int ai = threadID;
	int bi = threadID + (size / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
	temp[bi + bankOffsetB] = g_idata[blockOffset + bi];

	// build sum in place up the tree
	int offset = 1;
	for (int d = size >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) { 
		sums[blockID] = temp[size - 1 + CONFLICT_FREE_OFFSET(size - 1)];
		temp[size - 1 + CONFLICT_FREE_OFFSET(size - 1)] = 0;
	} 

	// traverse down tree & build scan
	for (int d = 1; d < size; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_odata[blockOffset + ai] = temp[ai + bankOffsetA];
	g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

namespace cuAlgo {

void exclusiveScan1dVector(int          *g_idata,
                           int          *g_odata,
                           unsigned int  size   ,
                           cudaStream_t  stream ,
                           bool          async  ) {

	unsigned int blocks = size / ELEMENTS_PER_BLOCK;
	int *d_sums, *d_incr;
	check_cuda( cudaMalloc(&d_sums, blocks * sizeof(int)) );
	check_cuda( cudaMalloc(&d_incr, blocks * sizeof(int)) );

	// Multi blocks
	{
		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(size, ELEMENTS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		unsigned int shmem = ELEMENTS_PER_BLOCK*sizeof(int);

		TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async,
		     exclusiveScan1dKernelMultiBlock,
		     g_idata, g_odata, d_sums, ELEMENTS_PER_BLOCK);
	}

	// Multi block (recursion) or single block
	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {

		exclusiveScan1dVector(d_sums, d_incr, blocks);
	} else {

		dim3 threadsPerBlock((blocks + 1) / 2);
		dim3 blocksPerGrid(1);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		unsigned int shmem = (blocks + 1) / 2 * sizeof(int);

		TIME( blocksPerGrid, threadsPerBlock, shmem, stream, async,
		      exclusiveScan1dKernelBlock,
		      d_sums, d_incr, blocks );
	}

	// Final step
	{

		dim3 threadsPerBlock(ELEMENTS_PER_BLOCK);
		dim3 blocksPerGrid(blocks);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
		      add,
		      g_odata, ELEMENTS_PER_BLOCK, d_incr );
	}

	check_cuda( cudaFree ( d_sums ) );
	check_cuda( cudaFree ( d_incr ) );
}
}