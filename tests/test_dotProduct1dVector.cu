/*
 * @file test_dotProduct1dVector.cu
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
#include "src/API/dotProduct1dVector.hpp"
#include <gtest/gtest.h>

TEST(dotProduct1dVector, default_value) {

	unsigned int nblocks = 4096;
	unsigned int size = 1024*nblocks;
	int * input1 = (int *)malloc(size * sizeof(int));
	int * input2 = (int *)malloc(size * sizeof(int));
	int * output = (int *)malloc(sizeof(int));
	int * solution = (int *)malloc(sizeof(int));

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
		input1[j + i*1024] = j;

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
		input2[j + i*1024] = j;

	int *d_input1;
	check_cuda( cudaMalloc(&d_input1, size*sizeof(int)) );

	int *d_input2;
	check_cuda( cudaMalloc(&d_input2, size*sizeof(int)) );

	int *d_output;
	check_cuda( cudaMalloc(&d_output, sizeof(int)) );

	check_cuda( cudaMemcpy ( d_input1, input1, (unsigned int)size*sizeof(int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_input2, input2, (unsigned int)size*sizeof(int), cudaMemcpyHostToDevice ) );

	cuAlgo::dotProduct1dVector<int, 1024, 2>(d_input1, d_input2, d_output, size);

	solution[0] = 0;
	for(unsigned int i = 0; i < size; ++i)
		solution[0] += input1[i] * input2[i];

	check_cuda( cudaMemcpy ( output, d_output, sizeof(int), cudaMemcpyDeviceToHost ) );

	ASSERT_EQ(solution[0], output[0]);

	check_cuda( cudaFree(d_input1) );
	check_cuda( cudaFree(d_input2) );
	check_cuda( cudaFree(d_output) );
	free(input1);
	free(input2);
	free(output);
	free(solution);
}