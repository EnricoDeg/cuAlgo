/*
 * @file convolution_reduction1dmatrix.cu
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
#include "cuAlgoInternal.hpp"
#include "utils.hpp"

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define COMPUTE_PER_THREAD   8

template <typename T>
__global__ void convolutionReduction1dMatrixKernel(const T *__restrict__ R,
                                                   const T *__restrict__ V,
                                                         T *__restrict__ C,
                                                   unsigned int          N,
                                                   unsigned int          K,
                                                   unsigned int     chunks) {

	const unsigned int tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

	T tmp1 = 0;
	T tmp2 = 0;
#pragma unroll
	for (unsigned int i = 0; i < K / chunks; ++i) {

		unsigned int col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const unsigned int row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			tmp1 += R[col + N * row] * V[col + N * row];
			tmp2 += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
		} else if (col > 0 && col < N /2) {

			tmp1 += R[col + N * row] * V[col + N * row] -
			        R[N - col + N * row] * V[N - col + N * row] ;
			tmp2 += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
			        R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
		}
	}

	unsigned int tidx = ( tid ) % ( N / 2 ) ;
	unsigned int tidy = ( tid ) / ( N / 2 ) ;
	C[tidx + N * tidy]         = tmp1;
	C[tidx + N / 2 + N * tidy] = tmp2;
}

template <typename T>
__global__ void convolutionReduction1dMatrixKernel1(const T *__restrict__ R,
                                                    const T *__restrict__ V,
                                                          T *__restrict__ C,
                                                    unsigned int          N,
                                                    unsigned int          K,
                                                    unsigned int     chunks) {

	__shared__ T sdata1[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];
	__shared__ T sdata2[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

	const unsigned int col = blockIdx.x * THREADS_PER_BLOCK_X + threadIdx.x;
	      unsigned int row = blockIdx.y * THREADS_PER_BLOCK_Y + threadIdx.y;

	if (col < N / 2 && row < chunks) {

		sdata1[threadIdx.y][threadIdx.x] = 0;
		sdata2[threadIdx.y][threadIdx.x] = 0;

		if (col == 0) {

#pragma unroll
			for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
				sdata1[threadIdx.y][threadIdx.x] += R[col         + N * row] * V[col         + N * row];
				sdata2[threadIdx.y][threadIdx.x] += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
			}
		} else if (col > 0 && col < N / 2) {

#pragma unroll
			for (unsigned int i = 0; i < K / chunks; ++i, row+=chunks) {
				sdata1[threadIdx.y][threadIdx.x] += R[col + N * row]         * V[col + N * row] -
				                                    R[N - col + N * row]     * V[N - col + N * row] ;
				sdata2[threadIdx.y][threadIdx.x] += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
				                                    R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
			}
		}
	}

	__syncthreads();

	for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
		if (threadIdx.y < s) {
			sdata1[threadIdx.y][threadIdx.x] += sdata1[threadIdx.y + s][threadIdx.x];
			sdata2[threadIdx.y][threadIdx.x] += sdata2[threadIdx.y + s][threadIdx.x];
		}
		__syncthreads();
	}

	if (threadIdx.y == 0) {
		C[col         + N * blockIdx.y] = sdata1[0][threadIdx.x];
		C[col + N / 2 + N * blockIdx.y] = sdata2[0][threadIdx.x];
	}
}

template<typename T>
void convolutionReduction1dMatrix(T            *R     ,
                                  T            *V     ,
                                  T            *C     ,
                                  unsigned int  N     ,
                                  unsigned int  K     ,
                                  cudaStream_t  stream,
                                  bool          async ) {

	unsigned int chunks = K / COMPUTE_PER_THREAD;

	if (chunks > THREADS_PER_BLOCK_Y) {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks / 32 *sizeof(T)) );

		dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
		dim3 blocksPerGrid(div_ceil(N / 2, THREADS_PER_BLOCK_X), div_ceil(chunks, THREADS_PER_BLOCK_Y));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     convolutionReduction1dMatrixKernel1<T>,
		     R, V, d_buffer, N, K, chunks);

		reduce1dMatrix<T>(d_buffer, C, N, chunks/32, stream, async);

		check_cuda( cudaFree ( d_buffer ) );
	} else if (chunks < THREADS_PER_BLOCK_Y && chunks > 1) {

		T * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(T)) );

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     convolutionReduction1dMatrixKernel<T>,
		     R, V, d_buffer, N, K, chunks);

		reduce1dMatrix<T>(d_buffer, C, N, chunks, stream, async);

		check_cuda( cudaFree ( d_buffer ) );
	} else {

		dim3 threadsPerBlock(THREADS_PER_BLOCK);
		dim3 blocksPerGrid(div_ceil(N, THREADS_PER_BLOCK));
		print_kernel_config(threadsPerBlock, blocksPerGrid);

		TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
		     convolutionReduction1dMatrixKernel<T>,
		     R, V, C, N, K, chunks);
	}
}

namespace cuAlgo {

	void convolutionReduction1dMatrixFloat(float        *R     ,
	                                       float        *V     ,
	                                       float        *C     ,
	                                       unsigned int  N     ,
	                                       unsigned int  K     ,
	                                       cudaStream_t  stream,
	                                       bool          async )
	{

		convolutionReduction1dMatrix<float>(R, V, C, N , K, stream, async);
	}

	void convolutionReduction1dMatrixDouble(double       *R     ,
	                                        double       *V     ,
	                                        double       *C     ,
	                                        unsigned int  N     ,
	                                        unsigned int  K     ,
	                                        cudaStream_t  stream,
	                                        bool          async )
	{

		convolutionReduction1dMatrix<double>(R, V, C, N , K, stream, async);
	}

	void convolutionReduction1dMatrixInt(int          *R     ,
	                                     int          *V     ,
	                                     int          *C     ,
	                                     unsigned int  N     ,
	                                     unsigned int  K     ,
	                                     cudaStream_t  stream,
	                                     bool          async )
	{

		convolutionReduction1dMatrix<int>(R, V, C, N , K, stream, async);
	}
}
