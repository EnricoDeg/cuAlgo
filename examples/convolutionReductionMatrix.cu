/*
 * @file convolutionReductionMatrix.cu
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
#include <stdlib.h>
#include <cuAlgo.hpp>
#include <chrono>

using namespace std::chrono;

int main() {

	size_t K = 8192;
	size_t N = 4096;

	int * R = (int *)malloc(K * N * sizeof(int));
	for (size_t i = 0 ; i < K ; ++i)
		for (size_t j = 0 ; j < N ; ++j)
			R [j + i * N] = j * i;

	int * V = (int *)malloc(K * N * sizeof(int));
	for (size_t i = 0 ; i < K ; ++i)
		for (size_t j = 0 ; j < N ; ++j)
			V [j + i * N] = N * K - j * i;

	int * C = (int *)malloc(N * sizeof(int));
	int * solution = (int *)malloc(N * sizeof(int));

	int *d_R;
	check_cuda( cudaMalloc(&d_R, K * N * sizeof(int)) );

	int *d_V;
	check_cuda( cudaMalloc(&d_V, K * N * sizeof(int)) );

	int *d_C;
	check_cuda( cudaMalloc(&d_C, N * sizeof(int)) );

	check_cuda( cudaMemcpy ( d_R, R, K * N *sizeof(int), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_V, V, K * N *sizeof(int), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	for (size_t i = 0; i < 5; ++i) {
		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrix(d_R, d_V, d_C, N, K);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function (GPU): " << duration.count() << " microseconds" << std::endl;
	}
	std::cout << "launching kernels done ..." << std::endl;

	for (size_t i = 0 ; i < N ; ++i)
		solution[i] = 0;

	auto start = high_resolution_clock::now();
	for (int j = 0 ; j < K ; ++j) {

		solution[0] += R[j * N] * V[j * N];

		for (int i = 1; i < N / 2; ++i)
			solution[i] += R[i + j * N] * V[i + j * N] -
			               R[N - i + j * N] * V[N - i + j * N];

		solution[N / 2] += R[N / 2 + j * N] * V[N / 2 + j * N];

		for (int i = N / 2 + 1, k = 0; i < N; ++i, ++k)
			solution[i] += R[N / 2 - 1 - k + j * N] * V[i + j * N] +
			               R[i + j * N] * V[N / 2 - 1 - k + j * N];
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function (CPU): " << duration.count() << " microseconds" << std::endl;

	check_cuda( cudaMemcpy ( C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost ) );

	for (size_t i = 0 ; i < N ; ++i)
		if (solution[i] != C[i]) {
			std::cout << "Values different" << std::endl;
			exit(EXIT_FAILURE);
		}

	return 0;
}
