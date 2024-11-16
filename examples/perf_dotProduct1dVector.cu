/*
 * @file perf_dotProduct1dVector.cu
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

	unsigned int nblocks = 4096;
	unsigned int size = 1024*nblocks;
	float * input1 = (float *)malloc(size * sizeof(float));
	float * input2 = (float *)malloc(size * sizeof(float));
	float * output = (float *)malloc(       sizeof(float));

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
		input1[j + i*1024] = j;

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
		input2[j + i*1024] = j;

	float *d_input1;
	check_cuda( cudaMalloc(&d_input1, size*sizeof(float)) );

	float *d_input2;
	check_cuda( cudaMalloc(&d_input2, size*sizeof(float)) );

	float *d_output;
	check_cuda( cudaMalloc(&d_output, sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input1, input1, (unsigned int)size*sizeof(float), cudaMemcpyHostToDevice ) );

	check_cuda( cudaMemcpy ( d_input2, input2, (unsigned int)size*sizeof(float), cudaMemcpyHostToDevice ) );

	for (unsigned int i = 0; i < 5; ++i)
		cuAlgo::dotProduct1dVectorFloat(d_input1, d_input2, d_output, size);

	check_cuda( cudaFree(d_input1) );
	check_cuda( cudaFree(d_input2) );
	check_cuda( cudaFree(d_output) );
	free(input1);
	free(input2);
	free(output);

	return 0;
}
