#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.h"

using namespace std::chrono;

#define THREADS_PER_BLOCK 1024
#define COMPUTE_PER_THREAD 128

__global__ void convolutionTaperReduction1dMatrixKernel(const int *__restrict__ R,
                                                        const int *__restrict__ V,
                                                        const int *__restrict__ T,
                                                              int *__restrict__ C,
                                                        size_t                  N,
                                                        size_t                  K,
                                                        size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

	int tmp1 = 0;
	int tmp2 = 0;
#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {

		size_t col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const size_t row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			tmp1 += T[row] * ( R[col + N * row] * V[col + N * row] );
			tmp2 += T[row] * ( R[col + N / 2 + N * row] * V[col + N / 2 + N * row] );
		} else if (col > 0 && col < N /2) {

			tmp1 += T[row] * ( R[col + N * row] * V[col + N * row] -
			                   R[N - col + N * row] * V[N - col + N * row] );
			tmp2 += T[row] * ( R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
			                   R[col + N / 2 + N * row] * V[N / 2 - col + N * row] );
		}
	}

	size_t tidx = ( tid ) % ( N / 2 ) ;
	size_t tidy = ( tid ) / ( N / 2 ) ;
	C[tidx + N * tidy]         = tmp1;
	C[tidx + N / 2 + N * tidy] = tmp2;
}

void convolutionTaperReduction1dMatrix(int *  R,
                                       int *  V,
                                       int *  T,
                                       int *  C,
                                       size_t N,
                                       size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	if (chunks > 1) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2 * chunks, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionTaperReduction1dMatrixKernel<<<grid, block>>>(R, V, T, d_buffer, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1dMatrix(d_buffer, C, N, chunks);

		check_cuda( cudaFree ( d_buffer ) );
	} else {

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionTaperReduction1dMatrixKernel<<<grid, block>>>(R, V, T, C, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	}
}
