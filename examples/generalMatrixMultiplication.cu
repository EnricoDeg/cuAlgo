#include <iostream>
#include <stdlib.h>
#include <cuAlgo.hpp>

int main() {

	cudaError_t err;
	int N = 1024 ;
	int M = N;
	int K = N;
	int T = 32;
	float alpha = 1.0f;
	float beta = 0.0f;

	float * A = (float *)malloc(N * N * sizeof(float));
	float * B = (float *)malloc(N * N * sizeof(float));
	float * C = (float *)malloc(N * N * sizeof(float));
	float * solution = (float *)malloc(N * N * sizeof(float));

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			A[j + i * N] = j * 1.0f;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			B[j + i * N] = i * 1.0f;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			C[j + i * N] = 1.0f;

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j)
			solution[j + i * N] = 0.0f;

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

	float *d_A;
	err = cudaMalloc(&d_A, N*N*sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	float *d_B;
	err = cudaMalloc(&d_B, N*N*sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	float *d_C;
	err = cudaMalloc(&d_C, N*N*sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_A, A, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_B, B, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_C, C, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 5; ++i)
		gMatMul(N, N, N, alpha, d_A, d_B, beta, d_C);

	err = cudaMemcpy ( C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < N; ++i)
		for (int j = 0; j < N ; ++j) {
			if ( fabsf( solution[j + i * N] - C[j + i * N] ) > 1.0e-6 ) {
				printf( "%d, %d, %f\n", i, j, fabsf( solution[j + i * N] - C[j + i * N] ) ) ;
			}
		}

	return 0;
}
