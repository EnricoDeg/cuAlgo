#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.h"

using namespace std::chrono;

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 32
#define THREADS_PER_BLOCK 1024  // WARP_SIZE * WARPS_PER_BLOCK
#define COMPUTE_PER_THREAD 128

__global__ void reduction1dMatrixKernel(const int *__restrict__ B,
                                              int *__restrict__ C,
                                        size_t                  N,
                                        size_t                  K,
                                        size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (tid > N * chunks)
		return;

	const size_t col   = ( blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ) % N;
	const size_t chunk = ( blockIdx.x * THREADS_PER_BLOCK + threadIdx.x ) / N;

	int tmp = 0;
	int offset = chunk * ( N * K / chunks );
#pragma unroll
	for (size_t i = 0; i < K/chunks; ++i) {
		tmp += B[i * N + col + offset];
	}
	C[col+chunk*N] = tmp;
}

void reduce1dMatrix(int *  B,
                    int *  C,
                    size_t N,
                    size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;

	if (chunks > 1) {

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
