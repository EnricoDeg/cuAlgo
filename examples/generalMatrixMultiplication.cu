#include <iostream>
#include <stdlib.h>
#include <cuAlgo.hpp>

int main() {

	cudaError_t err;
	int N = 1024 ;
	int M = N;
	int K = N;
	int T = 32;
	int alpha = 1;
	int beta = 0;

	int * A = (int *)malloc(N * N * sizeof(int));
	int * B = (int *)malloc(N * N * sizeof(int));
	int * C = (int *)malloc(N * N * sizeof(int));
	int * solution = (int *)malloc(N * N * sizeof(int));

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			A[j + i * N] = j * 1;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			B[j + i * N] = i * 1;

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
	err = cudaMalloc(&d_A, N*N*sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int *d_B;
	err = cudaMalloc(&d_B, N*N*sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int *d_C;
	err = cudaMalloc(&d_C, N*N*sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_A, A, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_B, B, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_C, C, (size_t)N*N*sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 5; ++i)
		gMatMul(N, N, N, alpha, d_A, d_B, beta, d_C);

	err = cudaMemcpy ( C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j) {
			if (  solution[j + i * N] != C[j + i * N] ) {
				printf( "%d, %d, %d, %d\n", i, j, solution[j + i * N] , C[j + i * N] ) ;
			}
		}

	return 0;
}
