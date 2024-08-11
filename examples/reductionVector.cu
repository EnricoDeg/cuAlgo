/*
 * @file reductionVector.cu
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

int main() {

	int nblocks = 4096;
	int size = 1024*nblocks;
	int * input = (int *)malloc(size * sizeof(int));
	int * output = (int *)malloc(sizeof(int));
	for(int i = 0; i < nblocks; ++i)
		for (int j = 0; j < 1024 ; ++j)
		input[j + i*1024] = j;

	int *d_input;
	check_cuda( cudaMalloc(&d_input, size*sizeof(int)) );

	int *d_output;
	check_cuda( cudaMalloc(&d_output, sizeof(int)) );

	check_cuda( cudaMemcpy ( d_input, input, (size_t)size*sizeof(int), cudaMemcpyHostToDevice ) );
	
	for (int i = 0; i < 5; ++i)
		reduce1dVector(d_input, d_output, size);

	output[0] = 0;
	for(int i = 0; i < size; ++i)
		output[0] += input[i];

	std::cout << "CPU solution = " << output[0] << std::endl;

	check_cuda( cudaMemcpy ( output, d_output, sizeof(int), cudaMemcpyDeviceToHost ) );

	std::cout << "GPU solution = " << output[0] << std::endl;

	return 0;
}
