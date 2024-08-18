/*
 * @file gSpMatVecMulELL.cu
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

#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void gSpMatVecMulELLKernel( const int * __restrict__ columns         ,
                                       const int * __restrict__ values          ,
                                       const int * __restrict__ x               ,
                                             int * __restrict__ y               ,
                                             int                nrows           ,
                                             int                elements_in_rows)
{

	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < nrows) {

		int sum = 0;
		for (size_t element = 0; element < elements_in_rows; ++element) {

			const unsigned int offset = row + element * nrows;
			sum += values[offset] * x[columns[offset]];
		}
		y[row] = sum;
	}
}

void gSpMatVecMulELL(int * columns         ,
                     int * values          ,
                     int * x               ,
                     int * y               ,
                     int   nrows           ,
                     int   elements_in_rows) {

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(div_ceil(nrows, THREADS_PER_BLOCK));

	std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
	std::cout << "blocksPerGrid   = " << div_ceil(nrows, THREADS_PER_BLOCK) << std::endl;

	auto start = high_resolution_clock::now();
	gSpMatVecMulELLKernel<<<grid, block>>>(columns, values, x, y, nrows, elements_in_rows);
	check_cuda( cudaDeviceSynchronize() );
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

}
