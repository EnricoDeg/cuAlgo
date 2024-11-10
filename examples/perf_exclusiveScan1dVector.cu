/*
 * @file perf_exclusiveScan1dVector.cu
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
#include <stdlib.h>
#include <cuAlgo.hpp>
#include <chrono>

int main() {

	const unsigned int size = 8192;
	float *idata = (float *)malloc(size * sizeof(float));
	float *odata = (float *)malloc(size * sizeof(float));

	for (unsigned int i = 0 ; i < size ; ++i) {
		idata[i]    = i * 2;
		odata[i]    = size - i;
	}

	float *d_idata;
	check_cuda( cudaMalloc(&d_idata, size * sizeof(float)) );
	float *d_odata;
	check_cuda( cudaMalloc(&d_odata, size * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_idata, idata, size * sizeof(float), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_odata, odata, size * sizeof(float), cudaMemcpyHostToDevice ) );

	for (unsigned int i = 0; i < 5; ++i) {
		std::cout << "launching kernels ..." << std::endl;
		cuAlgo::exclusiveScan1dVectorFloat(d_idata, d_odata, size);
		std::cout << "launching kernels done ..." << std::endl;
	}

	check_cuda( cudaFree(d_idata) );
	check_cuda( cudaFree(d_odata) );
	free(idata);
	free(odata);
}
