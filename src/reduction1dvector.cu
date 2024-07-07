#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

using namespace std::chrono;

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce1dKernel(int *g_idata, int *g_odata) {

	// use dynamic shared memory
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1dKernelFlexible(int *g_idata, int *g_odata) {

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

	int threadsPerBlock = size > 1024 ? 1024 : size/2;
	int blocksPerGrid = size / (2*threadsPerBlock) + (size % (2*threadsPerBlock) > 0);
	std::cout << "threadsPerBlock = " << threadsPerBlock << std::endl;
	std::cout << "blocksPerGrid   = " << blocksPerGrid   << std::endl;
	dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
	dim3 threadsPerBlock3(threadsPerBlock, 1, 1);

	if (blocksPerGrid == 1) {

		auto start = high_resolution_clock::now();
		reduce1dKernelFlexible<<<blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int)>>>(g_idata, g_odata);
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
		switch (threadsPerBlock) {
			case 1024:
			reduce1dKernel<1024><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 512:
			reduce1dKernel<512><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 256:
			reduce1dKernel<256><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 128:
			reduce1dKernel<128><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 64:
			reduce1dKernel< 64><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 32:
			reduce1dKernel< 32><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 16:
			reduce1dKernel< 16><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 8:
			reduce1dKernel< 8><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 4:
			reduce1dKernel< 4><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 2:
			reduce1dKernel< 2><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
			case 1:
			reduce1dKernel< 1><<< blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int) >>>(g_idata, d_buffer);
			break;
		}

		// reduce1dKernel<<<blocksPerGrid3, threadsPerBlock3, (size_t)threadsPerBlock*sizeof(int)>>>(g_idata, d_buffer);
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
