/*
 * @file test_upsample1dMatrix.cu
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
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include "src/cuAlgo.h"
#include <gtest/gtest.h>

using namespace std::chrono;

void upsample1dMatrix_CPU( float *idata, float *odata, unsigned int dim, int nzeros,
                           unsigned int mRows, unsigned int mCols ) {

	if (dim == 0) {

		for (size_t i = 0; i < mRows; ++i) {
			const float * __restrict in = idata + i*mCols;
			float * __restrict out = odata + i*(nzeros+1)*mCols;
			for (size_t j = 0; j < mCols; ++j)
				out[j] = in[j];
		}
	} else if (dim == 1) {

		size_t uCols = (mCols-1)*(nzeros)+mCols;

		for (size_t i = 0; i < mRows; ++i) {
			const float * __restrict in = idata + i*mCols;
			float * __restrict out = odata + i*uCols;
			for (size_t j = 0, k=0; j < mCols; ++j, k+=(nzeros+1))
				out[k] = in[j];
		}
	}
}

TEST(upsample1dMatrix, default_dim0) {

	unsigned int mRows  = 512;
	unsigned int mCols  = 512;
	unsigned int nzeros = 2;
	unsigned int mRowsUp = (mRows-1)*(nzeros)+mRows;

	float * idata    = (float *)malloc(mRows     * mCols * sizeof(float));
	for (unsigned int i = 0 ; i < mRows ; ++i)
		for (unsigned int j = 0 ; j < mCols ; ++j)
			idata[j + i * mCols] = j * i + 1;

	float * odata    = (float *)malloc(mRowsUp * mCols * sizeof(float));
	float * solution = (float *)malloc(mRowsUp * mCols * sizeof(float));

	float *d_idata;
	check_cuda( cudaMalloc(&d_idata, mRows   * mCols * sizeof(float)) );

	float *d_odata;
	check_cuda( cudaMalloc(&d_odata, mRowsUp * mCols * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::upsample1dMatrixFloat(d_idata, d_odata, 0, nzeros, mRows, mCols);

	for (unsigned int i = 0 ; i < mRowsUp * mCols ; ++i)
		solution[i] = 0;

	upsample1dMatrix_CPU(idata, solution, 0, nzeros, mRows, mCols);

	check_cuda( cudaMemcpy ( odata, d_odata, mRowsUp * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0 ; j < mRowsUp ; ++j)
		for (unsigned int i = 0 ; i < mCols ; ++i)
			ASSERT_EQ( solution[i + j * mCols] , odata[i + j * mCols] );

	check_cuda( cudaFree(d_idata) );
	check_cuda( cudaFree(d_odata) );
	free(idata);
	free(odata);
	free(solution);
}

TEST(upsample1dMatrix, default_dim1) {

	unsigned int mRows  = 512;
	unsigned int mCols  = 512;
	unsigned int nzeros = 2;
	unsigned int mColsUp = (mCols-1)*(nzeros)+mCols;

	float * idata    = (float *)malloc(mRows     * mCols * sizeof(float));
	for (unsigned int i = 0 ; i < mRows ; ++i)
		for (unsigned int j = 0 ; j < mCols ; ++j)
			idata[j + i * mCols] = j * i + 1;

	float * odata    = (float *)malloc(mRows * mColsUp * sizeof(float));
	float * solution = (float *)malloc(mRows * mColsUp * sizeof(float));

	float *d_idata;
	check_cuda( cudaMalloc(&d_idata, mRows * mCols   * sizeof(float)) );

	float *d_odata;
	check_cuda( cudaMalloc(&d_odata, mRows * mColsUp * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::upsample1dMatrixFloat(d_idata, d_odata, 1, nzeros, mRows, mCols);

	for (unsigned int i = 0 ; i < mRows * mColsUp ; ++i)
		solution[i] = 0;

	upsample1dMatrix_CPU(idata, solution, 1, nzeros, mRows, mCols);

	check_cuda( cudaMemcpy ( odata, d_odata, mRows * mColsUp * sizeof(float), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0 ; j < mRows ; ++j)
		for (unsigned int i = 0 ; i < mColsUp ; ++i)
			ASSERT_EQ( solution[i + j * mColsUp] , odata[i + j * mColsUp] );

	check_cuda( cudaFree(d_idata) );
	check_cuda( cudaFree(d_odata) );
	free(idata);
	free(odata);
	free(solution);

}

TEST(upsample1dMatrix, performance_dim0) {

	unsigned int mRows      = 512;
	unsigned int mCols      = 512;
	unsigned int nzeros     =   2;
	unsigned int iterations =  10;
	unsigned int mRowsUp = (mRows-1)*(nzeros)+mRows;

	float * idata    = (float *)malloc(mRows     * mCols * sizeof(float));
	for (unsigned int i = 0 ; i < mRows ; ++i)
		for (unsigned int j = 0 ; j < mCols ; ++j)
			idata[j + i * mCols] = j * i + 1;

	float *d_idata;
	check_cuda( cudaMalloc(&d_idata, mRows   * mCols * sizeof(float)) );

	float *d_odata;
	check_cuda( cudaMalloc(&d_odata, mRowsUp * mCols * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

	// warm-up
	cuAlgo::upsample1dMatrixFloat(d_idata, d_odata, 0, nzeros, mRows, mCols);

	auto start = high_resolution_clock::now();
	for (unsigned int i = 0; i < iterations; ++i)
		cuAlgo::upsample1dMatrixFloat(d_idata, d_odata, 0, nzeros, mRows, mCols);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << duration.count() / iterations << std::endl;
	ASSERT_TRUE(duration.count() / iterations < 25);

	check_cuda( cudaFree(d_idata) );
	check_cuda( cudaFree(d_odata) );
	free(idata);
}
