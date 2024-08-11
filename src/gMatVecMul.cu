/*
 * @file gMatVecMul.cu
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

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 2
#define COLS_PER_BLOCK 8  // COLS_PER_WARP * WARPS_PER_BLOCK
#define GROUP_SIZE 16     // WARP_SIZE / COLS_PER_WARP

__global__ void gMatVecMulKernel(const int *__restrict__ A,
                                 const int *__restrict__ B,
                                       int *__restrict__ C,
                                       size_t            N,
                                       size_t            K) {

	extern __shared__ int A_smem[];
	size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);

#pragma unroll
	for (size_t i = 0; i < A_smem_iters; ++i) {
		size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
		A_smem[idx] = A[idx];
	}

	__syncthreads();

	const size_t warp_id = threadIdx.x / WARP_SIZE;
	const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
	if (warp_col >= N)
		return;

	const size_t K_iters = div_ceil(K, WARP_SIZE);
	const size_t lane_id = threadIdx.x % WARP_SIZE;

	int tmp = 0.0;
#pragma unroll
	for (size_t i = 0; i < K_iters; ++i) {
		const size_t A_idx = i * WARP_SIZE + lane_id;
		const size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
		tmp += A_smem[A_idx] * B[B_idx];
	}

	const unsigned int mask = 0xffffffff;
#pragma unroll
	for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2)
		tmp += __shfl_xor_sync(mask, tmp, i);

	if (lane_id == 0)
		C[warp_col] = tmp;
}

__global__ void gMatVecMulKernel1(const int *__restrict__ A,
                                 const int *__restrict__ B,
                                       int *__restrict__ C,
                                       size_t            N,
                                       size_t            K) {

	extern __shared__ int A_smem[];
	size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);

#pragma unroll
	for (size_t i = 0; i < A_smem_iters; ++i) {
		size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
		A_smem[idx] = A[idx];
	}

	__syncthreads();

	const size_t group_id  = threadIdx.x / GROUP_SIZE;
	const size_t group_col = blockIdx.x * COLS_PER_BLOCK + group_id;
	if (group_col >= N)
		return;

	const size_t K_iters = div_ceil(K, GROUP_SIZE);
	const size_t group_lane_id = threadIdx.x % GROUP_SIZE;

	int tmp = 0.0;
#pragma unroll
	for (size_t i = 0; i < K_iters; ++i) {
		const size_t A_idx = i * GROUP_SIZE + group_lane_id;
		const size_t B_idx = i * GROUP_SIZE + group_lane_id + group_col * K;
		tmp += A_smem[A_idx] * B[B_idx];
	}

	constexpr unsigned int mask = 0xffffffff;
#pragma unroll
	for (size_t i = GROUP_SIZE / 2; i >= 1; i /= 2)
		tmp += __shfl_xor_sync(mask, tmp, i);

	if (group_lane_id == 0)
		C[group_col] = tmp;
}

void gMatVecMul(int *A, int *B, int *C, size_t N, size_t K) {

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(div_ceil(N, WARPS_PER_BLOCK));
	size_t smem = getSmem<int>(K);

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(N, THREADS_PER_BLOCK) << std::endl;

	auto start = high_resolution_clock::now();
	gMatVecMulKernel1<<<grid, block, smem>>>(A, B, C, N, K);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}
