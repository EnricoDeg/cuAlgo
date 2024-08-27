/*
 * @file gMatMul.cu
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
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

template <const uint BLOCKSIZE, typename T>
__global__ void gMatMulKernel(T                      alpha,
                              const T * __restrict__ A    ,
                              const T * __restrict__ B    ,
                              T                      beta ,
                              T       * __restrict__ C    ,
                              unsigned int           M    ,
                              unsigned int           N    ,
                              unsigned int           K    ) {

	// compute position in C that this thread is responsible for
	const unsigned int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
	const unsigned int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

	if (x < M && y < N) {
		int tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

template <typename T>
void gMatMul(T             alpha ,
             const T      *A     ,
             const T      *B     ,
             T             beta  ,
             T            *C     ,
             unsigned int  M     ,
             unsigned int  N     ,
             unsigned int  K     ,
             cudaStream_t  stream,
             bool          async ) {

	// create as many blocks as necessary to map all of C
	dim3 blocksPerGrid(div_ceil(M, 32), div_ceil(N, 32));
	dim3 threadsPerBlock(32 * 32);
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
	      gMatMulKernel<32 COMMA T>,
	      alpha, A, B, beta, C, M, N, K);
}

void gMatMulInt(int           alpha ,
                const int    *A     ,
                const int    *B     ,
                int           beta  ,
                int          *C     ,
                unsigned int  M     ,
                unsigned int  N     ,
                unsigned int  K     ,
                cudaStream_t  stream,
                bool          async )
{

	gMatMul<int>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
}

void gMatMulFloat(float         alpha ,
                  const float  *A     ,
                  const float  *B     ,
                  float         beta  ,
                  float        *C     ,
                  unsigned int  M     ,
                  unsigned int  N     ,
                  unsigned int  K     ,
                  cudaStream_t  stream,
                  bool          async )
{

	gMatMul<float>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
}

void gMatMulDouble(double        alpha ,
                   const double *A     ,
                   const double *B     ,
                   double        beta  ,
                   double       *C     ,
                   unsigned int  M     ,
                   unsigned int  N     ,
                   unsigned int  K     ,
                   cudaStream_t  stream,
                   bool          async )
{

	gMatMul<double>( alpha , A, B, beta, C, M, N, K, stream, async ) ;
}