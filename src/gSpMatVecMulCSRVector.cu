/*
 * @file gSpMatVecMulCSRVector.cu
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
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

#define FULL_WARP_MASK 0xffffffff

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 2
#define COLS_PER_BLOCK 8  // COLS_PER_WARP * WARPS_PER_BLOCK
#define GROUP_SIZE 16     // WARP_SIZE / COLS_PER_WARP

__device__ int warp_reduce(int val) {

	for (size_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
	return val;
}

__global__ void gSpMatVecMulCSRVectorKernel(const int * __restrict__ columns,
                                            const int * __restrict__ row_ptr,
                                            const int * __restrict__ values ,
                                            const int * __restrict__ x      ,
                                                  int * __restrict__ y      ,
                                                  int                nrows  ) {

	const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int warp_id   = thread_id / WARP_SIZE;
	const unsigned int lane      = thread_id % WARP_SIZE;

	const unsigned int row = warp_id;
	int sum = 0;
	if (row < nrows) {

		const unsigned int row_start = row_ptr[row    ];
		const unsigned int row_end   = row_ptr[row + 1];
		for (unsigned int element = row_start + lane; element < row_end; element+=WARP_SIZE) {
			sum += values[element] * x[columns[element]];
		}
	}
	sum = warp_reduce(sum);
	if (lane == 0 && row < nrows)
		y[row] = sum;
}
void gSpMatVecMulCSRVector(int * columns,
                           int * row_ptr,
                           int * values ,
                           int * x      ,
                           int * y      ,
                           int   nrows  ) {

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(div_ceil(nrows, WARPS_PER_BLOCK));

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(nrows, WARPS_PER_BLOCK) << std::endl;

	auto start = high_resolution_clock::now();
	gSpMatVecMulCSRVectorKernel<<<grid, block>>>(columns, row_ptr, values, x, y, nrows);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}
