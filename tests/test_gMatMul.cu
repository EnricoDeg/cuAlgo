/*
 * @file test_gMatMul.cu
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
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
#include "src/cuAlgo.hpp"
#include <gtest/gtest.h>

TEST(gMatMul, default_value) {

	int N     = 1024;
	int M     = N;
	int K     = N;
	int T     = 32;
	int alpha = 1;
	int beta  = 0;

	int * A        = (int *)malloc(N * N * sizeof(int));
	int * B        = (int *)malloc(N * N * sizeof(int));
	int * C        = (int *)malloc(N * N * sizeof(int));
	int * solution = (int *)malloc(N * N * sizeof(int));

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			A[j + i * N] = j;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			B[j + i * N] = i;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			C[j + i * N] = 1;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			solution[j + i * N] = 0;

	for (int m = 0; m < M; m += T) {
		for (int n = 0; n < N; n += T) {
			for (int k = 0; k < K; k += T) {

				const int minMt = std::min(m + T, M);
				const int minNt = std::min(n + T, N);
				const int minKt = std::min(k + T, K);

				for (int mt = m; mt < minMt; mt++) {
					for (int nt = n; nt < minNt; nt++) {
						for (int kt = k; kt < minKt; kt++) {
							solution[mt * M + nt] += A[mt * M + kt] * B[kt * K + nt];
						}
					}
				}
			}
		}
	}

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			solution[j + i * N] = alpha * solution[j + i * N] + beta * C[j + i * N];

	int *d_A;
	check_cuda( cudaMalloc(&d_A, N*N*sizeof(int)) );

	int *d_B;
	check_cuda( cudaMalloc(&d_B, N*N*sizeof(int)) );

	int *d_C;
	check_cuda( cudaMalloc(&d_C, N*N*sizeof(int)) );

	check_cuda( cudaMemcpy ( d_A, A, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_B, B, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_C, C, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice ) );

	for (int i = 0; i < 5; ++i)
		gMatMul(N, N, N, alpha, d_A, d_B, beta, d_C);

	check_cuda( cudaMemcpy ( C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost ) );

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j) {
			ASSERT_EQ(solution[j + i * N] , C[j + i * N]);
		}
}