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
#include "utils.hpp"

#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

template <typename T>
__global__ void transposeMatrixKernel(T *idata, T *odata,
                                      unsigned int width, unsigned int height)
{

	__shared__ T tile[TILE_DIM][TILE_DIM+1];
	unsigned int blockIdx_x, blockIdx_y;
	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}
	unsigned int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
	unsigned int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
	unsigned int index_out = xIndex + (yIndex)*height;
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}
	__syncthreads();
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		odata[index_out+i*height] =
		tile[threadIdx.x][threadIdx.y+i];
	}
}

template <typename T>
void transposeMatrix(T            *idata ,
                     T            *odata ,
                     unsigned int  size_x,
                     unsigned int  size_y,
                     cudaStream_t  stream,
                     bool          async ) {

	dim3 blocksPerGrid3(size_x / TILE_DIM, size_y / TILE_DIM, 1);
	dim3 threadsPerBlock3(TILE_DIM, BLOCK_ROWS, 1);

	print_kernel_config(threadsPerBlock3, blocksPerGrid3) ;

	TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
	     transposeMatrixKernel<T>,
	     idata, odata, size_x, size_y);
}

namespace cuAlgo {

	void transposeMatrixFloat(float        *idata ,
	                          float        *odata ,
	                          unsigned int  size_x,
	                          unsigned int  size_y,
	                          cudaStream_t  stream,
	                          bool          async ) {

		transposeMatrix<float>(idata, odata, size_x, size_y, stream, async);
	}

	void transposeMatrixDouble(double        *idata ,
	                           double        *odata ,
	                           unsigned int  size_x,
	                           unsigned int  size_y,
	                           cudaStream_t  stream,
	                           bool          async ) {

		transposeMatrix<double>(idata, odata, size_x, size_y, stream, async);
	}

	void transposeMatrixInt(int          *idata ,
	                        int          *odata ,
	                        unsigned int  size_x,
	                        unsigned int  size_y,
	                        cudaStream_t  stream,
	                        bool          async ) {

		transposeMatrix<int>(idata, odata, size_x, size_y, stream, async);
	}
}