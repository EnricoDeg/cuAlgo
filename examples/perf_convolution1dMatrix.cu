/*
 * @file perf_convolution1dMatrix.cu
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

	float * R = (float *)malloc(K * N * sizeof(float));
	for (unsigned int i = 0 ; i < K ; ++i)
		for (unsigned int j = 0 ; j < N ; ++j)
			R [j + i * N] = j * i;

	float * V = (float *)malloc(K * N * sizeof(float));
	for (unsigned int i = 0 ; i < K ; ++i)
		for (unsigned int j = 0 ; j < N ; ++j)
			V [j + i * N] = N * K - j * i;

	float *d_R;
	check_cuda( cudaMalloc(&d_R, K * N * sizeof(float)) );

	float *d_V;
	check_cuda( cudaMalloc(&d_V, K * N * sizeof(float)) );

	float *d_C;
	check_cuda( cudaMalloc(&d_C, N * K * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_R, R, K * N *sizeof(float), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_V, V, K * N *sizeof(float), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	for (unsigned int i = 0; i < 5; ++i)
		cuAlgo::convolution1dMatrixFloat(d_R, d_V, d_C, N, K);
	std::cout << "launching kernels done ..." << std::endl;

	check_cuda( cudaFree(d_R) );
	check_cuda( cudaFree(d_V) );
	check_cuda( cudaFree(d_C) );
	free(R);
	free(V);

	return 0;
}
