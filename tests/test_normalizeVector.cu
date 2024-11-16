/*
 * @file test_normalizeVector.cu
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
#include "src/cuAlgo.hpp"
#include <gtest/gtest.h>

TEST(normalizeVector, default_value) {

	unsigned int nblocks = 1024;
	unsigned int size = 1024*nblocks;

	float * input = (float *)malloc(size * sizeof(float));
	float * solution = (float *)malloc(size * sizeof(float));

	for(unsigned int i = 0; i < nblocks; ++i)
		for (unsigned int j = 0; j < 1024 ; ++j)
			input[j + i*1024] = -j - 1;

	float *d_input;
	check_cuda( cudaMalloc(&d_input, size*sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input, input, (unsigned int)size*sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::normalizeVectorFloat(d_input, size);

	float norm = 0;
	for(unsigned int i = 0; i < size; ++i)
		norm += std::abs(input[i]);

	for(unsigned int i = 0; i < size; ++i)
		solution[i] = input[i] / norm;

	check_cuda( cudaMemcpy ( input, d_input, (unsigned int)size*sizeof(float), cudaMemcpyDeviceToHost ) );

	for (unsigned int i = 0; i < size; ++i) 
		ASSERT_TRUE(std::abs(solution[i] - input[i]) / solution[i] < 1e-6);

	check_cuda( cudaFree(d_input ) );
	free(input);
	free(solution);
}