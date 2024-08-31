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

__global__ void exclusiveScan1dKernel(const int          *__restrict__ g_idata,
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

namespace cuAlgo {

void exclusiveScan1dVector(int          *g_idata,
                           int          *g_odata,
                           unsigned int  size   ,
                           cudaStream_t  stream ,
                           bool          async  ) {

	dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocksPerGrid(div_ceil(size/2, THREADS_PER_BLOCK));
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	unsigned int shmem = 2*THREADS_PER_BLOCK*sizeof(int);

	TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async,
	     exclusiveScan1dKernel,
	     g_idata, g_odata, size);
}
}