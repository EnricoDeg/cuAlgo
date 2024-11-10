/*
 * @file perf_gMatMul.cu
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

	unsigned int N = 1024;
	unsigned int M = N;
	unsigned int K = N;
	unsigned int T = 32;
	float alpha    = 1.0;
	float beta     = 0.0;

	float * A        = (float *)malloc(N * N * sizeof(float));
	float * B        = (float *)malloc(N * N * sizeof(float));
	float * C        = (float *)malloc(N * N * sizeof(float));
	float * solution = (float *)malloc(N * N * sizeof(float));

	for(unsigned int i = 0; i < N; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			A[j + i * N] = j;

	for(unsigned int i = 0; i < N; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			B[j + i * N] = i;

	for(unsigned int i = 0; i < N; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			C[j + i * N] = 1.0;

	for(unsigned int i = 0; i < N; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			solution[j + i * N] = 0.0;

	for(unsigned int m = 0; m < M; m += T) {
		for(unsigned int n = 0; n < N; n += T) {
			for(unsigned int k = 0; k < K; k += T) {

				const unsigned int minMt = std::min(m + T, M);
				const unsigned int minNt = std::min(n + T, N);
				const unsigned int minKt = std::min(k + T, K);

				for(unsigned int mt = m; mt < minMt; mt++) {
					for(unsigned int nt = n; nt < minNt; nt++) {
						for(unsigned int kt = k; kt < minKt; kt++) {
							solution[mt * M + nt] += A[mt * M + kt] * B[kt * K + nt];
						}
					}
				}
			}
		}
	}

	for(unsigned int i = 0; i < N; ++i)
		for(unsigned int j = 0; j < N ; ++j)
			solution[j + i * N] = alpha * solution[j + i * N] + beta * C[j + i * N];

	float *d_A;
	check_cuda( cudaMalloc(&d_A, N*N*sizeof(float)) );

	float *d_B;
	check_cuda( cudaMalloc(&d_B, N*N*sizeof(float)) );

	float *d_C;
	check_cuda( cudaMalloc(&d_C, N*N*sizeof(float)) );

	check_cuda( cudaMemcpy ( d_A, A, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_B, B, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_C, C, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice ) );

	for(unsigned int i = 0; i < 5; ++i)
		cuAlgo::gMatMulFloat(alpha, d_A, d_B, beta, d_C, N, N, N);

	check_cuda( cudaFree(d_A) );
	check_cuda( cudaFree(d_B) );
	check_cuda( cudaFree(d_C) );
	free(A);
	free(B);
	free(C);
	free(solution);

	return 0;
}
