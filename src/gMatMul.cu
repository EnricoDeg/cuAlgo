/*
 * @file gMatMul.cu
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

template <const uint BLOCKSIZE>
__global__ void gMatMulKernel(int M, int N, int K, int alpha, const int *A,
                              const int *B, int beta, int *C) {
	// compute position in C that this thread is responsible for
	const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
	const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);


	// `if` condition is necessary for when M or N aren't multiples of 32.
	if (x < M && y < N) {
		int tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		// C = α*(A@B)+β*C
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

void gMatMul(int M, int N, int K, int alpha, const int *A,
             const int *B, int beta, int *C) {

	// create as many blocks as necessary to map all of C
	dim3 blocksPerGrid3(div_ceil(M, 32), div_ceil(N, 32));
	// 32 * 32 = 1024 thread per block
	dim3 threadsPerBlock3(32 * 32);

	std::cout << "threadsPerBlock = " << 32 << ", " << 32 << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(M, 32) << ", " << div_ceil(N, 32) << std::endl;

	auto start = high_resolution_clock::now();
	gMatMulKernel<32><<< blocksPerGrid3, threadsPerBlock3 >>>(M, N, K, alpha, A, B, beta, C);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}