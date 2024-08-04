#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.h"

using namespace std::chrono;

__global__ void gMatMulKernel(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
	// compute position in C that this thread is responsible for
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// `if` condition is necessary for when M or N aren't multiples of 32.
	if (x < M && y < N) {
		float tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		// C = α*(A@B)+β*C
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

void gMatMul(int M, int N, int K, float alpha, const float *A,
             const float *B, float beta, float *C) {

	// create as many blocks as necessary to map all of C
	dim3 blocksPerGrid3(div_ceil(M, 32), div_ceil(N, 32), 1);
	// 32 * 32 = 1024 thread per block
	dim3 threadsPerBlock3(32, 32, 1);

	std::cout << "threadsPerBlock = " << 32 << ", " << 32 << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(M, 32) << ", " << div_ceil(N, 32) << std::endl;

	auto start = high_resolution_clock::now();
	gMatMulKernel<<< blocksPerGrid3, threadsPerBlock3 >>>(M, N, K, alpha, A, B, beta, C);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}