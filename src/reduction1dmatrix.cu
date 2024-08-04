#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.h"

using namespace std::chrono;

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define COMPUTE_PER_THREAD 64

__global__ void reduction1dMatrixKernel(const int *__restrict__ B,
                                              int *__restrict__ C,
                                        size_t                  N,
                                        size_t                  K,
                                        size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (tid > N * chunks)
		return;

	const size_t col   = tid % N;
	const size_t chunk = tid / N;
	const size_t tidm = col + chunk * N;

	int tmp = 0;
#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {
		tmp += B[i * N*chunks + tidm];
	}
	C[col+chunk*N] = tmp;
}

__global__ void reduction1dMatrixKernel1(const int *__restrict__ B,
                                               int *__restrict__ C,
                                         size_t                  N,
                                         size_t                  K,
                                         size_t             chunks) {

	const size_t tidx = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	const size_t tidy = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;
	if (tidx + N * tidy > N * chunks)
		return;

	const size_t tidm = tidx + tidy * N;

	int tmp = 0;
#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {
		tmp += B[i * N*chunks + tidm];
	}
	C[tidx+tidy*N] = tmp;
}

void reduce1dMatrix(int *  B,
                    int *  C,
                    size_t N,
                    size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	if (chunks > THREADS_PER_BLOCK_Y) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 grid(div_ceil(N, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK_X << ", "
		          << THREADS_PER_BLOCK_Y << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N, THREADS_PER_BLOCK_X) << ", "
		          << div_ceil(chunks, THREADS_PER_BLOCK_Y) << std::endl;

		auto start = high_resolution_clock::now();
		reduction1dMatrixKernel1<<<grid, block>>>(B, d_buffer, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1dMatrix(d_buffer, C, N, chunks);

		check_cuda( cudaFree ( d_buffer ) );
	} else if (chunks < THREADS_PER_BLOCK_Y && chunks > 1) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N, THREADS_PER_BLOCK)*chunks);

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N, THREADS_PER_BLOCK)*chunks << std::endl;

		auto start = high_resolution_clock::now();
		reduction1dMatrixKernel<<<grid, block>>>(B, d_buffer, N, K, chunks);
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
		std::cout << "blocksPerGrid   = " << div_ceil(N, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		reduction1dMatrixKernel<<<grid, block>>>(B, C, N, K, 1);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	}
}
