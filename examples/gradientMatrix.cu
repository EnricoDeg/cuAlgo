/*
 * @file gradientMatrix.cu
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

	unsigned int M = 4096;
	unsigned int N = 2048;

	int * A         = (int *)malloc(N * M * sizeof(int));
	int * Ax        = (int *)malloc(N * M * sizeof(int));
	int * Ay        = (int *)malloc(N * M * sizeof(int));
	int * solutionx = (int *)malloc(N * M * sizeof(int));
	int * solutiony = (int *)malloc(N * M * sizeof(int));

	for (unsigned int i = 0 ; i < N ; ++i)
		for (unsigned int j = 0 ; j < M ; ++j)
			A [j + i * M] = j + i * M;

	int *d_A;
	check_cuda( cudaMalloc(&d_A, M * N * sizeof(int)) );

	int *d_Ax;
	check_cuda( cudaMalloc(&d_Ax, M * N * sizeof(int)) );

	int *d_Ay;
	check_cuda( cudaMalloc(&d_Ay, M * N * sizeof(int)) );

	check_cuda( cudaMemcpy ( d_A, A, M * N *sizeof(int), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	for (unsigned int i = 0; i < 5; ++i)
		cuAlgo::gradMatrixInt(d_A, d_Ax, d_Ay, M, N);
	std::cout << "launching kernels done ..." << std::endl;

	for (unsigned int j = 0 ; j < M ; ++j)
		solutionx[j] = A[j];

	for (unsigned int i = 1 ; i < N ; ++i)
		for (unsigned int j = 0 ; j < M ; ++j)
			solutionx[j + i * M] = A[j + i * M] - A[j + (i - 1) * M];

	for (unsigned int i = 0 ; i < N ; ++i) {
		solutiony[i * M] = A[i * M];
		for (unsigned int j = 1 ; j < M ; ++j)
			solutiony[j + i * M] = A[j + i * M] - A[j - 1 + i * M];
	}

	check_cuda( cudaMemcpy ( Ax, d_Ax, M * N * sizeof(int), cudaMemcpyDeviceToHost ) );
	check_cuda( cudaMemcpy ( Ay, d_Ay, M * N * sizeof(int), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0 ; j < N ; ++j)
		for (unsigned int i = 0 ; i < M ; ++i)
			if (solutionx[i + j * M] != Ax[i + j * M]) {
				std::cout << "Values are different x" << std::endl;
				exit(EXIT_FAILURE);
			}

	for (unsigned int j = 0 ; j < N ; ++j)
		for (unsigned int i = 0 ; i < M ; ++i)
			if (solutiony[i + j * M] != Ay[i + j * M]) {
				std::cout << "Values are different y" << std::endl;
				exit(EXIT_FAILURE);
			}

	return 0;
}
