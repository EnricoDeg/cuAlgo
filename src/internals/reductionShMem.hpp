/*
 * @file reductionShMem.hpp
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

#ifndef REDUCTIONSHMEM_H
#define REDUCTIONSHMEM_H

#include <cuda.h>
#include "internals/operations.hpp"

template <unsigned int blockSize, typename T, template<typename> class op_t>
__device__ void warpReduceShMem(volatile T* sdata, unsigned int tid, op_t<T> &Op) {
	if (blockSize >= 64) Op.sharedMemory(&sdata[tid], &sdata[tid + 32]);
	if (blockSize >= 32) Op.sharedMemory(&sdata[tid], &sdata[tid + 16]);
	if (blockSize >= 16) Op.sharedMemory(&sdata[tid], &sdata[tid +  8]);
	if (blockSize >= 8)  Op.sharedMemory(&sdata[tid], &sdata[tid +  4]);
	if (blockSize >= 4)  Op.sharedMemory(&sdata[tid], &sdata[tid +  2]);
	if (blockSize >= 2)  Op.sharedMemory(&sdata[tid], &sdata[tid +  1]);
};

template <unsigned int blockSize, typename T, template<typename> class op_t>
__device__ void blockReduceShMemUnroll(volatile T* sdata, unsigned int tid, op_t<T> &Op) {

	if (blockSize >= 1024) {
		if (tid < 512) Op.sharedMemory(&sdata[tid], &sdata[tid + 512]);
		__syncthreads();
	}
	if (blockSize >= 512) {
		if (tid < 256) Op.sharedMemory(&sdata[tid], &sdata[tid + 256]);
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) Op.sharedMemory(&sdata[tid], &sdata[tid + 128]);
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid <  64) Op.sharedMemory(&sdata[tid], &sdata[tid +  64]);
		__syncthreads();
	}

	if (tid < 32) warpReduceShMem<blockSize, T, op_t>(sdata, tid, Op);
}

template <typename T, template<typename> class op_t>
__device__ void blockReduceShMem(volatile T* sdata, unsigned int tid, op_t<T> &Op) {

	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			Op.sharedMemory(&sdata[tid], &sdata[tid + s]);
		}
		__syncthreads();
	}
}

#endif