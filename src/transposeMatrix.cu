
#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

__global__ void transposeMatrixKernel(float *idata, float *odata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y) {

	dim3 blocksPerGrid3(size_x / TILE_DIM, size_y / TILE_DIM, 1);
	dim3 threadsPerBlock3(TILE_DIM, BLOCK_ROWS, 1);
	transposeMatrixKernel<<< blocksPerGrid3, threadsPerBlock3 >>>(idata, odata);
	cudaError_t err = cudaDeviceSynchronize();
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}