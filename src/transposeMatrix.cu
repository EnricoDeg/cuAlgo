
#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

using namespace std::chrono;

#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

__global__ void transposeMatrixKernel(float *idata, float *odata,
                                      unsigned int width, unsigned int height)
{

	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		tile[threadIdx.y+i][threadIdx.x] =
		idata[index_in+i*width];
	}
	__syncthreads();
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		odata[index_out+i*height] =
		tile[threadIdx.x][threadIdx.y+i];
		}
}

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y) {

	dim3 blocksPerGrid3(size_x / TILE_DIM, size_y / TILE_DIM, 1);
	dim3 threadsPerBlock3(TILE_DIM, BLOCK_ROWS, 1);

	std::cout << "threadsPerBlock = " << TILE_DIM << ", " << BLOCK_ROWS << std::endl;
	std::cout << "blocksPerGrid   = " << size_x / TILE_DIM << ", " << size_y / TILE_DIM << std::endl;

	auto start = high_resolution_clock::now();
	transposeMatrixKernel<<< blocksPerGrid3, threadsPerBlock3 >>>(idata, odata, size_x, size_y);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
}