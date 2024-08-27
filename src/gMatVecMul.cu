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
#include "utils.hpp"
#include "templateShMem.hpp"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 2
#define COLS_PER_BLOCK 8  // COLS_PER_WARP * WARPS_PER_BLOCK
#define GROUP_SIZE 16     // WARP_SIZE / COLS_PER_WARP

template <typename T>
__global__ void gMatVecMulKernel(const T          * __restrict__ A,
                                 const T          * __restrict__ B,
                                       T          * __restrict__ C,
                                       unsigned int                N,
                                       unsigned int                K) {

	// use dynamic shared memory
	// needed for template
	SharedMemory<T> smem;
	T * A_smem = smem.getPointer();

	unsigned int A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);

#pragma unroll
	for (unsigned int i = 0; i < A_smem_iters; ++i) {
		unsigned int idx = i * THREADS_PER_BLOCK + threadIdx.x;
		A_smem[idx] = A[idx];
	}

	__syncthreads();

	const unsigned int warp_id = threadIdx.x / WARP_SIZE;
	const unsigned int warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
	if (warp_col >= N)
		return;

	const unsigned int K_iters = div_ceil(K, WARP_SIZE);
	const unsigned int lane_id = threadIdx.x % WARP_SIZE;

	T tmp = 0.0;
#pragma unroll
	for (unsigned int i = 0; i < K_iters; ++i) {
		const unsigned int A_idx = i * WARP_SIZE + lane_id;
		const unsigned int B_idx = i * WARP_SIZE + lane_id + warp_col * K;
		tmp += A_smem[A_idx] * B[B_idx];
	}

	const unsigned int mask = 0xffffffff;
#pragma unroll
	for (unsigned int i = WARP_SIZE / 2; i >= 1; i /= 2)
		tmp += __shfl_xor_sync(mask, tmp, i);

	if (lane_id == 0)
		C[warp_col] = tmp;
}

template<typename T>
__global__ void gMatVecMulKernel1(const T          * __restrict__ A,
                                  const T          * __restrict__ B,
                                        T          * __restrict__ C,
                                        unsigned int                N,
                                        unsigned int                K) {

	// use dynamic shared memory
	// needed for template
	SharedMemory<T> smem;
	T * A_smem = smem.getPointer();

	unsigned int A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);

#pragma unroll
	for (unsigned int i = 0; i < A_smem_iters; ++i) {
		unsigned int idx = i * THREADS_PER_BLOCK + threadIdx.x;
		A_smem[idx] = A[idx];
	}

	__syncthreads();

	const unsigned int group_id  = threadIdx.x / GROUP_SIZE;
	const unsigned int group_col = blockIdx.x * COLS_PER_BLOCK + group_id;
	if (group_col >= N)
		return;

	const unsigned int K_iters = div_ceil(K, GROUP_SIZE);
	const unsigned int group_lane_id = threadIdx.x % GROUP_SIZE;

	T tmp = 0.0;
#pragma unroll
	for (unsigned int i = 0; i < K_iters; ++i) {
		const unsigned int A_idx = i * GROUP_SIZE + group_lane_id;
		const unsigned int B_idx = i * GROUP_SIZE + group_lane_id + group_col * K;
		tmp += A_smem[A_idx] * B[B_idx];
	}

	constexpr unsigned int mask = 0xffffffff;
#pragma unroll
	for (unsigned int i = GROUP_SIZE / 2; i >= 1; i /= 2)
		tmp += __shfl_xor_sync(mask, tmp, i);

	if (group_lane_id == 0)
		C[group_col] = tmp;
}

template <typename T>
void gMatVecMul(const T            *A     ,
                const T            *B     ,
                      T            *C     ,
                      unsigned int  N     ,
                      unsigned int  K     ,
                      cudaStream_t  stream,
                      bool          async ) {

	dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocksPerGrid(div_ceil(N, WARPS_PER_BLOCK));
	print_kernel_config(threadsPerBlock, blocksPerGrid);
	unsigned int smem = getSmem<T>(K);

	TIME( blocksPerGrid, threadsPerBlock, smem, stream, async, 
	      gMatVecMulKernel1,
	      A, B, C, N, K );
}

namespace cuAlgo {

	void gMatVecMulInt(const int          *A     ,
	                   const int          *B     ,
	                         int          *C     ,
	                         unsigned int  N     ,
	                         unsigned int  K     ,
	                         cudaStream_t  stream,
	                         bool          async )
	{

		gMatVecMul<int>(A, B, C, N, K, stream, async);
	}

	void gMatVecMulFloat(const float        *A      ,
	                     const float        *B      ,
	                           float        *C      ,
	                           unsigned int  N      ,
	                           unsigned int  K      ,
	                           cudaStream_t  stream ,
	                           bool          async  )
	{

		gMatVecMul<float>(A, B, C, N, K, stream, async);
	}

	void gMatVecMulDouble(const double       *A     ,
	                      const double       *B     ,
	                            double       *C     ,
	                            unsigned int  N     ,
	                            unsigned int  K     ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		gMatVecMul<double>(A, B, C, N, K, stream, async);
	}
}
