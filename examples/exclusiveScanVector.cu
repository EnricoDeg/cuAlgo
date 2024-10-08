/*
 * @file exclusiveScanVector.cu
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

using namespace std::chrono;

int main() {

	const unsigned int size = 8192;
	int *idata    = (int *)malloc(size * sizeof(int));
	int *odata    = (int *)malloc(size * sizeof(int));
	int *solution = (int *)malloc(size * sizeof(int));

	for (unsigned int i = 0 ; i < size ; ++i) {

		idata[i]    = i * 2;
		odata[i]    = size - i;
		solution[i] = size - i;
	}

	solution[0] = 0;
	for (unsigned int i = 1 ; i < size ; ++i) {

		solution[i] = idata[i-1] + solution[i-1];
	}

	int *d_idata;
	check_cuda( cudaMalloc(&d_idata, size * sizeof(int)) );
	int *d_odata;
	check_cuda( cudaMalloc(&d_odata, size * sizeof(int)) );

	check_cuda( cudaMemcpy ( d_idata, idata, size * sizeof(int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_odata, odata, size * sizeof(int), cudaMemcpyHostToDevice ) );

	std::cout << "launching kernels ..." << std::endl;
	cuAlgo::exclusiveScan1dVectorInt(d_idata, d_odata, size);
	std::cout << "launching kernels done ..." << std::endl;

	check_cuda( cudaMemcpy ( odata, d_odata, size * sizeof(int), cudaMemcpyDeviceToHost ) );

	for (unsigned int i = 0 ; i < size ; ++i) {
		if (solution[i] != odata[i]) {
			std::cout << "Values are different " << i << " - " << solution[i] << " - " << odata[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	free(idata);
	free(odata);
	free(solution);
}
