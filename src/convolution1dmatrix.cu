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

#define THREADS_PER_BLOCK 1024
#define COMPUTE_PER_THREAD  32

__global__ void convolution1dMatrixKernel(const int *__restrict__ R,
                                          const int *__restrict__ V,
                                                int *__restrict__ C,
                                          size_t                  N,
                                          size_t                  K) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (tid > N * K)
		return;

	const size_t col  = tid % N;
	const size_t row  = tid / N;

	if (col == 0 || col == N/2) {

		C[col + N * row] = R[col + N * row] * V[col + N * row];
	} else if (col > 0 && col < N /2) {

		C[col + N * row] = R[col + N * row] * V[col + N * row] -
		                   R[N - col + N * row] * V[N - col + N * row] ;
	} else {

		size_t j = col - ( N / 2 + 1 );
		C[col + N * row] = R[N / 2 - 1 - j + N * row] * V[col + N * row] +
		                   R[col + N * row] * V[N / 2 - 1 - j + N * row] ;
	}

}

__global__ void convolution1dMatrixKernel1(const int *__restrict__ R,
                                           const int *__restrict__ V,
                                                 int *__restrict__ C,
                                           size_t                  N,
                                           size_t                  K,
                                           size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {

		size_t col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const size_t row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			C[col + N * row] = R[col + N * row] * V[col + N * row];
			C[col + N / 2 + N * row] = R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
		} else if (col > 0 && col < N /2) {

			C[col + N * row] = R[col + N * row] * V[col + N * row] -
			                   R[N - col + N * row] * V[N - col + N * row] ;
			col += N / 2;
			size_t j = col - ( N / 2 + 1 );
			C[col + N * row] = R[N / 2 - 1 - j + N * row] * V[col + N * row] +
			                   R[col + N * row] * V[N / 2 - 1 - j + N * row] ;
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

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(N * chunks, THREADS_PER_BLOCK) << std::endl;

	auto start = high_resolution_clock::now();
	convolution1dMatrixKernel1<<<grid, block>>>(R, V, C, N, K, chunks);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}
