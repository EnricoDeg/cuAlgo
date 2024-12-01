/*
 * @file test_ifftshift2dMatrix.cu
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
#include <chrono>
#include "src/cuAlgo.h"
#include <gtest/gtest.h>

using namespace std::chrono;

void ifftshiftMatrixCPU(float * idata, float * odata,
                        unsigned int mRows, unsigned int mCols) {

	if (mRows % 2 == 0 && mCols % 2 == 0) {

		unsigned int hRows = mRows / 2;
		unsigned int hCols = mCols / 2;
		for (unsigned int i = 0; i < hRows; ++i) {

			float * __restrict in = idata +  (hRows + i) * mCols + hCols;
			float * __restrict out = odata + i*mCols;
			memcpy(out, in, hCols * sizeof(float));
			in -= mCols;
			memcpy(out + hCols, in + hCols, hCols * sizeof(float));
		}

		for (unsigned int i = hRows; i < mRows; ++i) {

			float * __restrict in = idata +  (i - hRows) * mCols + hCols ;
			float * __restrict out = odata + i*mCols;
			memcpy(out, in, hCols * sizeof(float));
			in -= mCols;
			memcpy(out + hCols, in + hCols, hCols * sizeof(float));
		}
	}
}

TEST(ifftshift2dMatrix, default_even_even) {

	unsigned int mRows = 1024;
	unsigned int mCols = 1024;

	float * input    = (float *)malloc(mRows * mCols * sizeof(float));
	float * solution = (float *)malloc(mRows * mCols * sizeof(float));

	for(unsigned int i = 0; i < mRows; ++i)
		for (unsigned int j = 0; j < mCols ; ++j)
		input[j + i*mCols] = i * j + 1;

	float *d_input;
	check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::ifftshift2dMatrixFloat(d_input, mRows, mCols);

	ifftshiftMatrixCPU(input, solution, mRows, mCols);

	check_cuda( cudaMemcpy ( input, d_input, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

	for(unsigned int i = 0; i < mRows; ++i)
		for (unsigned int j = 0; j < mCols ; ++j)
			ASSERT_EQ( input[j + i * mCols] , solution[j + i * mCols] );

	check_cuda( cudaFree(d_input) );
	free(input);
	free(solution);
}
