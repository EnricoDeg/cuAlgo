/*
 * @file test_downsample1dMatrix.cu
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
#include "src/cuAlgo.h"
#include <gtest/gtest.h>

void downsample1dMatrix_CPU( float * idata, float * odata,
                             unsigned int dim, int stride,
                             unsigned int mRows, unsigned int mCols ) {

    size_t count = 0;
    if (dim == 0) {
        for (size_t i = 0; i < mRows; i+=stride, ++count);
    } else if (dim == 1) {
        for (size_t i = 0; i < mCols; i+=stride, ++count);
    }

    if (dim == 0) {

        if (stride == 1) {

            std::memcpy(odata, idata, mRows*mCols*sizeof(float));
        } else {

            for (size_t i = 0; i < count; ++i) {
                const float * __restrict in = idata + (i*stride)*mCols;
                float * __restrict out = odata + i*mCols;
                for (size_t j = 0; j < mCols; ++j)
                    out[j] = in[j];
            }
        }
    } else if (dim == 1) {

        if (stride == 1) {

            std::memcpy(odata, idata, mRows*mCols*sizeof(float));
        } else {

            for (size_t i = 0; i < mRows; ++i) {
                const float * __restrict in = idata + i*mCols;
                float * __restrict out = odata + i*count;
                for (size_t j = 0; j < count; ++j)
                    out[j] = in[j*stride];
            }
        }
    }
}

TEST(downsample1dMatrix, default_values_dim0) {

	unsigned int mRows  = 1024;
	unsigned int mCols  = 1024;
	unsigned int stride = 2;
	unsigned int mRowsDown = mRows / stride;

	float * idata    = (float *)malloc(mRows     * mCols * sizeof(float));
	for (unsigned int i = 0 ; i < mRows ; ++i)
		for (unsigned int j = 0 ; j < mCols ; ++j)
			idata[j + i * mCols] = j * i + 1;

	float * odata    = (float *)malloc(mRowsDown * mCols * sizeof(float));
	float * solution = (float *)malloc(mRowsDown * mCols * sizeof(float));

	float *d_idata;
	check_cuda( cudaMalloc(&d_idata, mRows     * mCols * sizeof(float)) );

	float *d_odata;
	check_cuda( cudaMalloc(&d_odata, mRowsDown * mCols * sizeof(float)) );

	check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

	cuAlgo::downsample1dMatrixFloat(d_idata, d_odata, 0, stride, mRows, mCols);

	for (unsigned int i = 0 ; i < mRowsDown * mCols ; ++i)
		solution[i] = 0;

	downsample1dMatrix_CPU(idata, solution, 0, stride, mRows, mCols);

	check_cuda( cudaMemcpy ( odata, d_odata, mRowsDown * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0 ; j < mRowsDown ; ++j)
		for (unsigned int i = 0 ; i < mCols ; ++i)
			ASSERT_EQ( solution[i + j * mCols] , odata[i + j * mCols] );

	check_cuda( cudaFree(d_idata) );
	check_cuda( cudaFree(d_odata) );
	free(idata);
	free(odata);
	free(solution);
}
