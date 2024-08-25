/*
 * @file convolution_reduction1dmatrix.cu
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
#define COMPUTE_PER_THREAD   8

__global__ void convolutionReduction1dMatrixKernel(const int *__restrict__ R,
                                                   const int *__restrict__ V,
                                                         int *__restrict__ C,
                                                   size_t                  N,
                                                   size_t                  K,
                                                   size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

	int tmp1 = 0;
	int tmp2 = 0;
#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {

		size_t col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const size_t row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			tmp1 += R[col + N * row] * V[col + N * row];
			tmp2 += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
		} else if (col > 0 && col < N /2) {

			tmp1 += R[col + N * row] * V[col + N * row] -
			        R[N - col + N * row] * V[N - col + N * row] ;
			tmp2 += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
			        R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
		}
	}

	size_t tidx = ( tid ) % ( N / 2 ) ;
	size_t tidy = ( tid ) / ( N / 2 ) ;
	C[tidx + N * tidy]         = tmp1;
	C[tidx + N / 2 + N * tidy] = tmp2;
}

__global__ void convolutionReduction1dMatrixKernel1(const int *__restrict__ R,
                                                    const int *__restrict__ V,
                                                          int *__restrict__ C,
                                                    size_t                  N,
                                                    size_t                  K,
                                                    size_t             chunks) {

	__shared__ int sdata1[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];
	__shared__ int sdata2[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

	const size_t col = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	      size_t row = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;

	if (col < N / 2 && row < chunks) {

		sdata1[threadIdx.y][threadIdx.x] = 0;
		sdata2[threadIdx.y][threadIdx.x] = 0;

		if (col == 0) {

#pragma unroll
			for (size_t i = 0; i < K / chunks; ++i, row+=chunks) {
				sdata1[threadIdx.y][threadIdx.x] += R[col         + N * row] * V[col         + N * row];
				sdata2[threadIdx.y][threadIdx.x] += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
			}
		} else if (col > 0 && col < N / 2) {

#pragma unroll
			for (size_t i = 0; i < K / chunks; ++i, row+=chunks) {
				sdata1[threadIdx.y][threadIdx.x] += R[col + N * row]         * V[col + N * row] -
				                                    R[N - col + N * row]     * V[N - col + N * row] ;
				sdata2[threadIdx.y][threadIdx.x] += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
				                                    R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
			}
		}
	}

	__syncthreads();

	for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
		if (threadIdx.y < s) {
			sdata1[threadIdx.y][threadIdx.x] += sdata1[threadIdx.y + s][threadIdx.x];
			sdata2[threadIdx.y][threadIdx.x] += sdata2[threadIdx.y + s][threadIdx.x];
		}
		__syncthreads();
	}

	if (threadIdx.y == 0) {
		C[col         + N * blockIdx.y] = sdata1[0][threadIdx.x];
		C[col + N / 2 + N * blockIdx.y] = sdata2[0][threadIdx.x];
	}
}

void convolutionReduction1dMatrix(int *  R,
                                  int *  V,
                                  int *  C,
                                  size_t N,
                                  size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	if (chunks > THREADS_PER_BLOCK_Y) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks / 32 *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 grid(div_ceil(N / 2, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK_X << ", "
		          << THREADS_PER_BLOCK_Y << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2, THREADS_PER_BLOCK_X) << ", "
		          << div_ceil(chunks, THREADS_PER_BLOCK_Y) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrixKernel1<<<grid, block>>>(R, V, d_buffer, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1dMatrixInt(d_buffer, C, N, chunks/32, 0, false);

		check_cuda( cudaFree ( d_buffer ) );
	} else if (chunks < THREADS_PER_BLOCK_Y && chunks > 1) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2 * chunks, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrixKernel<<<grid, block>>>(R, V, d_buffer, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1dMatrixInt(d_buffer, C, N, chunks, 0, false);

		check_cuda( cudaFree ( d_buffer ) );
	} else {

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrixKernel<<<grid, block>>>(R, V, C, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	}
}
