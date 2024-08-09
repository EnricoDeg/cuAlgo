#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.h"

using namespace std::chrono;

#define THREADS_PER_BLOCK 1024
#define COMPUTE_PER_THREAD  32

__global__ void convolution1dMatrixKernel(const int *__restrict__ R,
                                          const int *__restrict__ V,
                                                int *__restrict__ C,
                                          size_t                  N,
                                          size_t                  K) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (tid > N * K)
		return;

	const size_t col  = tid % N;
	const size_t row  = tid / N;

	if (col == 0 || col == N/2) {

		C[col + N * row] = R[col + N * row] * V[col + N * row];
	} else if (col > 0 && col < N /2) {

		C[col + N * row] = R[col + N * row] * V[col + N * row] -
		                   R[N - col + N * row] * V[N - col + N * row] ;
	} else {

		size_t j = col - ( N / 2 + 1 );
		C[col + N * row] = R[N / 2 - 1 - j + N * row] * V[col + N * row] +
		                   R[col + N * row] * V[N / 2 - 1 - j + N * row] ;
	}

}

__global__ void convolution1dMatrixKernel1(const int *__restrict__ R,
                                           const int *__restrict__ V,
                                                 int *__restrict__ C,
                                           size_t                  N,
                                           size_t                  K,
                                           size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {

		size_t col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const size_t row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			C[col + N * row] = R[col + N * row] * V[col + N * row];
			C[col + N / 2 + N * row] = R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
		} else if (col > 0 && col < N /2) {

			C[col + N * row] = R[col + N * row] * V[col + N * row] -
			                   R[N - col + N * row] * V[N - col + N * row] ;
			col += N / 2;
			size_t j = col - ( N / 2 + 1 );
			C[col + N * row] = R[N / 2 - 1 - j + N * row] * V[col + N * row] +
			                   R[col + N * row] * V[N / 2 - 1 - j + N * row] ;
		}
	}
}

void convolution1dMatrix(int *  R,
                         int *  V,
                         int *  C,
                         size_t N,
                         size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(N * chunks, THREADS_PER_BLOCK) << std::endl;

	auto start = high_resolution_clock::now();
	convolution1dMatrixKernel1<<<grid, block>>>(R, V, C, N, K, chunks);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}
