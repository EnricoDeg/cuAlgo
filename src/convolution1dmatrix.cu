/*
 * @file convolution1dmatrix.cu
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
#include <chrono>
#include "cuAlgo.hpp"
#include "utils.hpp"

using namespace std::chrono;

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define COMPUTE_PER_THREAD  4

template <typename T>
__global__ void convolution1dMatrixKernel(const T *__restrict__ R,
                                          const T *__restrict__ V,
                                                T *__restrict__ C,
                                          unsigned int          N,
                                          unsigned int          K,
                                          unsigned int     chunks) {

	const unsigned int col = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	      unsigned int row = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;

	if (col < N / 2 && row < chunks) {

		if (col == 0) {

#pragma unroll
			for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
				C[col         + N * row] = R[col + N * row]         * V[col + N * row];
				C[col + N / 2 + N * row] = R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
			}
		} else if (col > 0 && col < N / 2) {

#pragma unroll
			for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
				C[col         + N * row] = R[col + N * row]         * V[col + N * row] -
				                           R[N - col + N * row]     * V[N - col + N * row] ;
				C[col + N / 2 + N * row] = R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
				                           R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
			}
		}
	}
}

template<typename T>
void convolution1dMatrix(T            *R     ,
                         T            *V     ,
                         T            *C     ,
                         unsigned int  N     ,
                         unsigned int  K     ,
                         cudaStream_t  stream,
                         bool          async ) {

	unsigned int chunks = K / COMPUTE_PER_THREAD;

	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 blocksPerGrid(div_ceil(N / 2, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
	     convolution1dMatrixKernel<T>,
	     R, V, C, N, K, chunks);
}

namespace cuAlgo {

	void convolution1dMatrixFloat(float        *R     ,
	                              float        *V     ,
	                              float        *C     ,
	                              unsigned int  N     ,
	                              unsigned int  K     ,
	                              cudaStream_t  stream,
	                              bool          async )
	{

		convolution1dMatrix<float>(R, V, C, N, K, stream, async);
	}

	void convolution1dMatrixDouble(double       *R     ,
	                               double       *V     ,
	                               double       *C     ,
	                               unsigned int  N     ,
	                               unsigned int  K     ,
	                               cudaStream_t  stream,
	                               bool          async )
	{

		convolution1dMatrix<double>(R, V, C, N, K, stream, async);
	}

	void convolution1dMatrixInt(int          *R     ,
	                            int          *V     ,
	                            int          *C     ,
	                            unsigned int  N     ,
	                            unsigned int  K     ,
	                            cudaStream_t  stream,
	                            bool          async )
	{

		convolution1dMatrix<int>(R, V, C, N, K, stream, async);
	}
}
