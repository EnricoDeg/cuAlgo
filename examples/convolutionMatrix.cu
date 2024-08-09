#include <iostream>
#include <stdlib.h>
#include <cuAlgo.hpp>

int main() {

	cudaError_t err;
	size_t K = 1024;
	size_t N = 1024;

	int * R = (int *)malloc(K * N * sizeof(int));
	for (size_t i = 0 ; i < K ; ++i)
		for (size_t j = 0 ; j < N ; ++j)
			R [j + i * N] = j * i;

	int * V = (int *)malloc(K * N * sizeof(int));
	for (size_t i = 0 ; i < K ; ++i)
		for (size_t j = 0 ; j < N ; ++j)
			V [j + i * N] = N * K - j * i;

	int * C = (int *)malloc(N*K * sizeof(int));
	int * solution = (int *)malloc(N*K * sizeof(int));

	int *d_R;
	err = cudaMalloc(&d_R, K * N * sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int *d_V;
	err = cudaMalloc(&d_V, K * N * sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int *d_C;
	err = cudaMalloc(&d_C, N * K * sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_R, R, K * N *sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_V, V, K * N *sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "launching kernels ..." << std::endl;
	for (size_t i = 0; i < 5; ++i)
		convolution1dMatrix(d_R, d_V, d_C, N, K);
	std::cout << "launching kernels done ..." << std::endl;

	for (size_t i = 0 ; i < N ; ++i)
		solution[i] = 0;

	for (int j = 0 ; j < K ; ++j) {

		solution[j * N] = R[j * N] * V[j * N];

		for (int i = 1; i < N / 2; ++i)
			solution[i + j * N] = R[i + j * N] * V[i + j * N] -
			                      R[N - i + j * N] * V[N - i + j * N];

		solution[N / 2 + j * N] = R[N / 2 + j * N] * V[N / 2 + j * N];

		for (int i = N / 2 + 1, k = 0; i < N; ++i, ++k)
			solution[i + j * N] = R[N / 2 - 1 - k + j * N] * V[i + j * N] +
			                      R[i + j * N] * V[N / 2 - 1 - k + j * N];
	}

	err = cudaMemcpy ( C, d_C, N * K * sizeof(int), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMemcpy): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t j = 0 ; j < K ; ++j)
		for (size_t i = 0 ; i < N ; ++i)
			if (solution[i + j * N] != C[i + j * N]) {
				std::cout << "Values different" << std::endl;
				exit(EXIT_FAILURE);
			}

	return 0;
}
