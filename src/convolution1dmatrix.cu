/*
 * @file convolution1dmatrix.cu
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
#include <chrono>
#include "cuAlgo.hpp"
#include "utils.hpp"

using namespace std::chrono;

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define COMPUTE_PER_THREAD  4

__global__ void convolution1dMatrixKernel(const int *__restrict__ R,
                                          const int *__restrict__ V,
                                                int *__restrict__ C,
                                          size_t                  N,
                                          size_t                  K,
                                          size_t             chunks) {

	const size_t col = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	      size_t row = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;

	if (col < N / 2 && row < chunks) {

		if (col == 0) {

#pragma unroll
			for (size_t i = 0; i < K / chunks; ++i, row+=chunks) {
				C[col         + N * row] = R[col + N * row]         * V[col + N * row];
				C[col + N / 2 + N * row] = R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
			}
		} else if (col > 0 && col < N / 2) {

#pragma unroll
			for (size_t i = 0; i < K / chunks; ++i, row+=chunks) {
				C[col         + N * row] = R[col + N * row]         * V[col + N * row] -
				                           R[N - col + N * row]     * V[N - col + N * row] ;
				C[col + N / 2 + N * row] = R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
				                           R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
			}
		}
	}
}

void convolution1dMatrix(int *  R,
                         int *  V,
                         int *  C,
                         size_t N,
                         size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 grid(div_ceil(N / 2, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK_X << ", " <<
	                                     THREADS_PER_BLOCK_Y << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(N / 2 , THREADS_PER_BLOCK_X) << ", " <<
	                                     div_ceil(chunks, THREADS_PER_BLOCK_Y) << std::endl;

	auto start = high_resolution_clock::now();
	convolution1dMatrixKernel<<<grid, block>>>(R, V, C, N, K, chunks);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}
