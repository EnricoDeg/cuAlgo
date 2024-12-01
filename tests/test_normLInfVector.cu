/*
 * @file test_normLInfVector.cu
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
#include "src/cuAlgo.h"
#include <gtest/gtest.h>

TEST(normLInfVector, default_value) {

	unsigned int nblocks = 1024;
	unsigned int size = 1024*nblocks;
	float * input = (float *)malloc(size * sizeof(float));
	float * output = (float *)malloc(sizeof(float));
	float * solution = (float *)malloc(sizeof(float));
	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
			input[j + i*1024] = -j;

	float *d_input;
	check_cuda( cudaMalloc(&d_input, size*sizeof(float)) );

	float *d_output;
	check_cuda( cudaMalloc(&d_output, sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input, input, (unsigned int)size*sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::normLInfVectorFloat(d_input, d_output, size);

	solution[0] = 0;
	for(unsigned int i = 0; i < size; ++i)
		solution[0] = max(solution[0], std::abs(input[i]));

	check_cuda( cudaMemcpy ( output, d_output, sizeof(float), cudaMemcpyDeviceToHost ) );

	ASSERT_TRUE(std::abs(solution[0] - output[0]) / solution[0] < 1e-6);

	check_cuda( cudaFree(d_input ) );
	check_cuda( cudaFree(d_output) );
	free(input);
	free(output);
	free(solution);
}