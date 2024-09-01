/*
 * @file gradientMatrix.cu
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

#include "cuAlgo.hpp"
#include "utils.hpp"

template<typename T>
__global__ void gradMatrixKernel(const T            *__restrict__ A ,
                                       T            *__restrict__ Ax,
                                       T            *__restrict__ Ay,
                                       unsigned int               M ,
                                       unsigned int               N ) {

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < M && y < N) {

		if (y == 0)
			Ax[x + y * M] = A[x + y * M];
		else
			Ax[x + y * M] = A[x + y * M] - A[x + (y - 1) * M];

		if (x == 0)
			Ay[x + y * M] = A[x + y * M];
		else
			Ay[x + y * M] = A[x + y * M] - A[x - 1 + y * M];
	}
}

template <typename T>
void gradMatrix(T            *A ,
                T            *Ax,
                T            *Ay,
                unsigned int  M,
                unsigned int  N,
                cudaStream_t  stream,
                bool          async ) {

	dim3 blocksPerGrid(div_ceil(M, 32), div_ceil(N, 32));
	dim3 threadsPerBlock(32 , 32);
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
	      gradMatrixKernel<T>,
	      A, Ax, Ay, M, N);
}

namespace cuAlgo {

	void gradMatrixInt(int          *A     ,
	                   int          *Ax    ,
	                   int          *Ay    ,
	                   unsigned int  M     ,
	                   unsigned int  N     ,
	                   cudaStream_t  stream,
	                   bool          async )
	{

		gradMatrix<int>(A , Ax, Ay, M, N, stream, async );
	}

	void gradMatrixFloat(float        *A     ,
	                     float        *Ax    ,
	                     float        *Ay    ,
	                     unsigned int  M     ,
	                     unsigned int  N     ,
	                     cudaStream_t  stream,
	                     bool          async )
	{

		gradMatrix<float>(A , Ax, Ay, M, N, stream, async );
	}

	void gradMatrixDouble(double       *A     ,
	                      double       *Ax    ,
	                      double       *Ay    ,
	                      unsigned int  M     ,
	                      unsigned int  N     ,
	                      cudaStream_t  stream,
	                      bool          async )
	{

		gradMatrix<double>(A , Ax, Ay, M, N, stream, async );
	}
}