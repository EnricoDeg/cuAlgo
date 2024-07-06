#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

using namespace std::chrono;

__global__ void reduce1dKernel(int *g_idata, int *g_odata) {

	// use dynamic shared memory
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce1d(int *g_idata, int *g_odata, int size) {

	int threadsPerBlock = size > 1024 ? 512 : size;
	int blocksPerGrid = size / threadsPerBlock + (size % threadsPerBlock > 0);
	std::cout << "threadsPerBlock = " << threadsPerBlock << std::endl;
	std::cout << "blocksPerGrid   = " << blocksPerGrid   << std::endl;
	dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
	dim3 threadsPerBlock3(threadsPerBlock, 1, 1);

	if (blocksPerGrid == 1) {

		auto start = high_resolution_clock::now();
		reduce1dKernel<<<blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int)>>>(g_idata, g_odata);
		cudaError_t err = cudaDeviceSynchronize();
		if ( err != cudaSuccess ) {
			std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	} else {

		int * d_buffer;
		cudaError_t err = cudaMalloc(&d_buffer, blocksPerGrid*sizeof(int));
		if (err != cudaSuccess) {
			std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}
		auto start = high_resolution_clock::now();
		reduce1dKernel<<<blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int)>>>(g_idata, d_buffer);
		err = cudaDeviceSynchronize();
		if ( err != cudaSuccess ) {
			std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1d(d_buffer, g_odata, blocksPerGrid);
		err = cudaFree ( d_buffer );
		if ( err != cudaSuccess ) {
			fprintf(stderr, "CUDA error (cudaFree): %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
}
