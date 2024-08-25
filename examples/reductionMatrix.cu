/*
 * @file reductionMatrix.cu
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

int main() {

	unsigned int K = 8192;
	unsigned int N = 4096;

	int * B = (int *)malloc(K * N * sizeof(int));
	for (size_t i = 0 ; i < K ; ++i)
		for (size_t j = 0 ; j < N ; ++j)
			B [j + i * N] = j*i;

	int * C = (int *)malloc(N * sizeof(int));
	int * solution = (int *)malloc(N * sizeof(int));

	int *d_B;
	check_cuda( cudaMalloc(&d_B, K * N * sizeof(int)) );

	int *d_C;
	check_cuda( cudaMalloc(&d_C, N * sizeof(int)) );

	check_cuda( cudaMemcpy ( d_B, B, K * N *sizeof(int), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	for (size_t i = 0; i < 5; ++i)
		cuAlgo::reduce1dMatrixInt(d_B, d_C, N, K);
	std::cout << "launching kernels done ..." << std::endl;

	for (size_t i = 0 ; i < N ; ++i)
		solution[i] = 0;

	for (int i = 0 ; i < K ; ++i)
		for (int j = 0 ; j < N ; ++j)
			solution[j] += B [j + i * N];

	check_cuda( cudaMemcpy ( C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost ) );

	for (size_t i = 0 ; i < N ; ++i)
		if (solution[i] != C[i]) {
			std::cout << "Values are different" << std::endl;
			exit(EXIT_FAILURE);
		}

	return 0;
}
