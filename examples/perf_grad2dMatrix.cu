/*
 * @file perf_gradMatrix.cu
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

	float * A         = (float *)malloc(N * M * sizeof(float));
	float * Ax        = (float *)malloc(N * M * sizeof(float));
	float * Ay        = (float *)malloc(N * M * sizeof(float));
	float * solutionx = (float *)malloc(N * M * sizeof(float));
	float * solutiony = (float *)malloc(N * M * sizeof(float));

	for (unsigned int i = 0 ; i < N ; ++i)
		for (unsigned int j = 0 ; j < M ; ++j)
			A [j + i * M] = j + i * M;

	float *d_A;
	check_cuda( cudaMalloc(&d_A , M * N * sizeof(float)) );

	float *d_Ax;
	check_cuda( cudaMalloc(&d_Ax, M * N * sizeof(float)) );

	float *d_Ay;
	check_cuda( cudaMalloc(&d_Ay, M * N * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_A, A, M * N *sizeof(float), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	for (unsigned int i = 0; i < 5; ++i)
		cuAlgo::grad2dMatrixFloat(d_A, d_Ax, d_Ay, M, N);
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

	check_cuda( cudaFree(d_A) );
	check_cuda( cudaFree(d_Ax) );
	check_cuda( cudaFree(d_Ay) );
	free(A);
	free(Ax);
	free(Ay);
	free(solutionx);
	free(solutiony);

	return 0;
}
