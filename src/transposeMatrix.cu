/*
 * @file transposeMatrix.cu
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <iostream>
#include "cuAlgo.hpp"
#include <chrono>

using namespace std::chrono;

#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

__global__ void transposeMatrixKernel(float *idata, float *odata,
                                      unsigned int width, unsigned int height)
{

	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	int blockIdx_x, blockIdx_y;
	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}
	int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
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