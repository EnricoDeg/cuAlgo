/*
 * @file test_taper1dMatrix.cu
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

TEST(taper1dMatrix, default_value) {

	unsigned int M = 4096;
	unsigned int N = 2048;
	unsigned int taperLength = 32;

	int          * A            = (         int *)malloc(N           * M * sizeof(         int));
	int          * taper        = (         int *)malloc(taperLength *     sizeof(         int));
	unsigned int * startIndices = (unsigned int *)malloc(N           *     sizeof(unsigned int));
	unsigned int * endIndices   = (unsigned int *)malloc(N           *     sizeof(unsigned int));
	int          * solution     = (         int *)malloc(N           * M * sizeof(         int));

	for (unsigned int i = 0 ; i < N ; ++i)
		for (unsigned int j = 0 ; j < M ; ++j) {
			A       [j + i * M] = j + i * M;
			solution[j + i * M] = j + i * M;
		}

	for (unsigned int i = 0 ; i < N ; ++i) {
		startIndices[i] =     64;
		endIndices  [i] = M - 64;
	}

	for (unsigned int i = 0 ; i < taperLength ; ++i)
		taper[i] = i;

	int *d_A;
	check_cuda( cudaMalloc(&d_A           , M           * N * sizeof(         int)) );

	int *d_taper;
	check_cuda( cudaMalloc(&d_taper       , taperLength *     sizeof(         int)) );

	unsigned int *d_startIndices;
	check_cuda( cudaMalloc(&d_startIndices, N           *     sizeof(unsigned int)) );

	unsigned int *d_endIndices;
	check_cuda( cudaMalloc(&d_endIndices  , N           *     sizeof(unsigned int)) );

	check_cuda( cudaMemcpy ( d_A           , A           , M           * N * sizeof(         int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_taper       , taper       , taperLength *     sizeof(         int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_startIndices, startIndices, N           *     sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_endIndices  , endIndices  , N           *     sizeof(unsigned int), cudaMemcpyHostToDevice ) );

	cuAlgo::taper1dMatrixInt(d_A, d_taper, d_startIndices, d_endIndices, M, N, taperLength);

	check_cuda( cudaMemcpy ( A, d_A, M * N * sizeof(int), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0 ; j < N ; ++j) {

		for (unsigned int i = 0; i < startIndices[j]; ++i)
			solution[i + j * M] = 0;

		for (unsigned int i = startIndices[j]; i < startIndices[j]+taperLength ; ++i)
			solution[i + j * M] *= taper[i-startIndices[j]];

		for (unsigned int i = endIndices[j] - taperLength + 1 ; i < endIndices[j] + 1 ; ++i)
			solution[i + j * M] *= taper[ taperLength - 1 - ( i - ( endIndices[j] - taperLength + 1 ) ) ];

		for (unsigned int i = endIndices[j]+1; i < M ; ++i)
			solution[i + j * M] = 0;
	}

	for (unsigned int j = 0 ; j < N ; ++j)
		for (unsigned int i = 0 ; i < M ; ++i)
			ASSERT_EQ(solution[i + j * M], A[i + j * M]);

	check_cuda( cudaFree(d_A) );
	check_cuda( cudaFree(d_taper) );
	check_cuda( cudaFree(d_startIndices) );
	check_cuda( cudaFree(d_endIndices) );
	free(A);
	free(taper);
	free(startIndices);
	free(endIndices);
	free(solution);
}
