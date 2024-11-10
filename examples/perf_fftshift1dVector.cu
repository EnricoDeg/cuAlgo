/*
 * @file perf_fftshift1dVector.cu
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

#define BLOCKSIZE 1024

int main() {

	unsigned int nblocks = 2;
	unsigned int size = BLOCKSIZE * nblocks;

	float * input  = (float *)malloc(size * sizeof(float));

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < BLOCKSIZE ; ++j)
		input[j + i*BLOCKSIZE] = j;

	float *d_input;
	check_cuda( cudaMalloc(&d_input , size * sizeof(float)) );

	float *d_output;
	check_cuda( cudaMalloc(&d_output, size * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input, input, (unsigned int)size * sizeof(float), cudaMemcpyHostToDevice ) );

	for (unsigned int i = 0; i < 5; ++i)
		cuAlgo::fftshift1dVectorFloat(d_input, d_output, size);

	check_cuda( cudaFree(d_input) );
	check_cuda( cudaFree(d_output) );
	free(input);

	return 0;
}
