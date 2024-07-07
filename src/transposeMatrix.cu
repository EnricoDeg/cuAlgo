
#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

using namespace std::chrono;

#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

__global__ void transposeMatrixKernel(float *idata, float *odata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

}

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y) {

	dim3 blocksPerGrid3(size_x / TILE_DIM, size_y / TILE_DIM, 1);
	dim3 threadsPerBlock3(TILE_DIM, BLOCK_ROWS, 1);

	std::cout << "threadsPerBlock = " << TILE_DIM << ", " << BLOCK_ROWS << std::endl;
	std::cout << "blocksPerGrid   = " << size_x / TILE_DIM << ", " << size_y / TILE_DIM << std::endl;

	auto start = high_resolution_clock::now();
	transposeMatrixKernel<<< blocksPerGrid3, threadsPerBlock3 >>>(idata, odata);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}