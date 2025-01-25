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
#include "cuAlgo.h"
#include "internals/utils.hpp"

template <const uint BLOCKSIZE, typename T>
__global__ void gMatMulKernel(T                      alpha,
                              const T * __restrict__ A    ,
                              const T * __restrict__ B    ,
                              T                      beta ,
                              T       * __restrict__ C    ,
                              unsigned int           M    ,
                              unsigned int           N    ,
                              unsigned int           K    ) {

    // output matrix block in this thread block
    const unsigned int cRow = blockIdx.x;
    const unsigned int cCol = blockIdx.y;

    __shared__ T As[BLOCKSIZE * BLOCKSIZE];
    __shared__ T Bs[BLOCKSIZE * BLOCKSIZE];

    // inner row and col in block
    const unsigned int threadCol = (threadIdx.x % BLOCKSIZE);
    const unsigned int threadRow = (threadIdx.x / BLOCKSIZE);

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    T tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];

}

namespace cuAlgo {

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

	template void gMatMul(float , const float  *, const float  *,
	                      float , float  *,
	                      unsigned int, unsigned int, unsigned int,
	                      cudaStream_t, bool);
	template void gMatMul(double, const double *, const double *,
	                      double, double *,
	                      unsigned int, unsigned int, unsigned int,
	                      cudaStream_t, bool);
}
