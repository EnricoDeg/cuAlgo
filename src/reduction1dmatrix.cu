/*
 * @file reduction1dmatrix.cu
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
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define COMPUTE_PER_THREAD 128

template <typename T>
__global__ void reduction1dMatrixKernel(const T *__restrict__ B,
                                              T *__restrict__ C,
                                        unsigned int          N,
                                        unsigned int          K,
                                        unsigned int     chunks) {

	const unsigned int tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (tid > N * chunks)
		return;

	const unsigned int col   = tid % N;
	const unsigned int chunk = tid / N;
	const unsigned int tidm = col + chunk * N;

	T tmp = 0;
#pragma unroll
	for (unsigned int i = 0; i < K / chunks; ++i) {
		tmp += B[i * N*chunks + tidm];
	}
	C[col+chunk*N] = tmp;
}

template <typename T>
__global__ void reduction1dMatrixKernel1(const T *__restrict__ B,
                                               T *__restrict__ C,
                                         unsigned int          N,
                                         unsigned int          K,
                                         unsigned int     chunks) {

	const unsigned int tidx = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int tidy = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;
	if (tidx + N * tidy > N * chunks)
		return;

	const unsigned int tidm = tidx + tidy * N;

	T tmp = 0;
#pragma unroll
	for (unsigned int i = 0; i < K / chunks; ++i) {
		tmp += B[i * N*chunks + tidm];
	}
	C[tidx+tidy*N] = tmp;
}

template<typename T>
__global__ void reduction1dMatrixKernel2(const T *__restrict__ B,
                                               T *__restrict__ C,
                                         unsigned int          N,
                                         unsigned int          K,
                                         unsigned int     chunks) {

	__shared__ T sdata[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

	const unsigned int tidx = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int tidy = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;
	if (tidx + N * tidy > N * chunks)
		return;

	const unsigned int tidm = tidx + tidy * N;

	sdata[threadIdx.y][threadIdx.x] = 0;
#pragma unroll
	for (unsigned int i = 0; i < K / chunks; ++i) {
		sdata[threadIdx.y][threadIdx.x] += B[i * N*chunks + tidm];
	}

	__syncthreads();

	for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
		if (threadIdx.y < s) {
			sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y + s][threadIdx.x];
		}
		__syncthreads();
	}

	if (threadIdx.y == 0)
		C[tidx+blockIdx.y*N] = sdata[0][threadIdx.x];
}

template <typename T>
void reduce1dMatrix(T            *B     ,
                    T            *C     ,
                    unsigned int  N     ,
                    unsigned int  K     ,
                    cudaStream_t  stream,
                    bool          async ) {

	unsigned int chunks = K / COMPUTE_PER_THREAD;

	if (chunks > THREADS_PER_BLOCK_Y) {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks / 32 *sizeof(T)) );

		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(N, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     reduction1dMatrixKernel2<T>,
		     B, d_buffer, N, K, chunks);

		reduce1dMatrix<T>(d_buffer, C, N, chunks/32, stream, async);

		check_cuda( cudaFree ( d_buffer ) );
	} else if (chunks < THREADS_PER_BLOCK_Y && chunks > 1) {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(T)) );

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(N, THREADS_PER_BLOCK)*chunks);
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     reduction1dMatrixKernel<T>,
		     B, d_buffer, N, K, chunks);

		reduce1dMatrix<T>(d_buffer, C, N, chunks, stream, async);

		check_cuda( cudaFree ( d_buffer ) );
	} else {

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(N, THREADS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     reduction1dMatrixKernel<T>,
		     B, C, N, K, 1);
	}
}

namespace cuAlgo {

	void reduce1dMatrixInt(int *         B     ,
	                       int *         C     ,
	                       unsigned int  N     ,
	                       unsigned int  K     ,
	                       cudaStream_t  stream,
	                       bool          async ) {

		reduce1dMatrix<int>(B, C, N , K, stream, async);
	}

	void reduce1dMatrixFloat(float *         B     ,
	                         float *         C     ,
	                         unsigned int  N     ,
	                         unsigned int  K     ,
	                         cudaStream_t  stream,
	                         bool          async ) {

		reduce1dMatrix<float>(B, C, N , K, stream, async);
	}

	void reduce1dMatrixDouble(double *         B     ,
	                          double *         C     ,
	                          unsigned int  N     ,
	                          unsigned int  K     ,
	                          cudaStream_t  stream,
	                          bool          async ) {

		reduce1dMatrix<double>(B, C, N , K, stream, async);
	}
}