/*
 * @file perf_gMatVecMul.cu
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

	unsigned int N = 8192;
	unsigned int K = N;

	float * B        = (float *)malloc(N * K * sizeof(float));
	float * A        = (float *)malloc(K     * sizeof(float));
	float * C        = (float *)malloc(N     * sizeof(float));
	float * solution = (float *)malloc(N     * sizeof(float));

	for(unsigned int i = 0; i < K; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			B[j + i * N] = j*i;

	for(unsigned int j = 0; j < K ; ++j)
		A[j] = j;

	float *d_B;
	check_cuda( cudaMalloc(&d_B, N * K * sizeof(float)) );

	float *d_A;
	check_cuda( cudaMalloc(&d_A, K *     sizeof(float)) );

	float *d_C;
	check_cuda( cudaMalloc(&d_C, N *     sizeof(float)) );

	check_cuda( cudaMemcpy ( d_B, B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_A, A, (size_t)K     * sizeof(float), cudaMemcpyHostToDevice ) );

	for(unsigned int i = 0; i < 5; ++i)
		cuAlgo::gMatVecMulFloat(d_A, d_B, d_C, N, K);

	for(unsigned int j = 0; j < N ; ++j)
		solution[j] = 0;

	for(unsigned int i = 0; i < K; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			solution[j] += A[i] * B[j + i * K];

	check_cuda( cudaFree(d_A) );
	check_cuda( cudaFree(d_B) );
	check_cuda( cudaFree(d_C) );
	free(A);
	free(B);
	free(C);
	free(solution);
}
