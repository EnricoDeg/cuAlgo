/*
 * @file transpositionMatrix.cu
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

	unsigned int size_x = 1024;
	unsigned int size_y = 1024;
	float * input  = (float *)malloc(size_x * size_y * sizeof(float));
	float * output = (float *)malloc(size_x * size_y * sizeof(float));
	for(unsigned int j = 0; j < size_y; ++j)
		for (unsigned int i = 0; i < size_x ; ++i)
		input[i + j * size_x] = i + j * size_x;

	float *d_input;
	check_cuda( cudaMalloc(&d_input, size_x * size_y * sizeof(float)) );

	float *d_output;
	check_cuda( cudaMalloc(&d_output, size_x * size_y * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input, input, size_x * size_y * sizeof(float), cudaMemcpyHostToDevice ) );

	for (int i = 0; i < 5; ++i)
		transposeMatrix(d_input, d_output, size_x, size_y);

	check_cuda( cudaMemcpy ( output, d_output, size_x * size_y * sizeof(float), cudaMemcpyDeviceToHost ) );

	for(int j = 0; j < size_y; ++j)
		for (int i = 0; i < size_x ; ++i)
			if (input[i + j * size_x] != output[j + i * size_y]) {
				std::cout << "Values different" << std::endl;
				return 1;
			}

	return 0;
}
