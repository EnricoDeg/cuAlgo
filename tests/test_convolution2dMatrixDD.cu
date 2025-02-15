/*
 * @file test_convolution2dMatrixDD.cu
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
#include <gtest/gtest.h>
#include "cuAlgo/API/convolution2dMatrixDD.hpp"

void convolution2DMatrixCC_CPU(float * result, float * data, float * filter,
                               unsigned int mRows, unsigned int mCols,
                               unsigned int fRows, unsigned int fCols) {

    unsigned int rRows = mRows + fRows - 1;
    unsigned int rCols = mCols + fCols - 1;

    for (unsigned int nr = 0; nr < rRows; ++nr) {
        unsigned int low_mr = std::max((long int)0, (long int)nr - (long int)fRows + 1);
        unsigned int high_mr = std::min((long int)mRows - 1, (long int)nr);
        for (unsigned int nc = 0; nc < rCols; ++nc) {
            unsigned int low_mc = std::max((long int)0, (long int)nc - (long int)fCols + 1);
            unsigned int high_mc = std::min((long int)mCols - 1 , (long int)nc);
            float tmp = 0.0;
            for (unsigned int mr = low_mr; mr <= high_mr; ++mr)
                for (unsigned int mc = low_mc; mc <= high_mc; ++mc)
                    tmp += data[mr * rCols + mc] * filter[(nr-mr)*rCols+(nc-mc)];
            result[nr * rCols + nc] = tmp;
        }
    }
}

TEST(convolution2dMatrixDD, default_values) {

    unsigned int mRows = 512;
    unsigned int mCols = 512;
    unsigned int fRows = 32;
    unsigned int fCols = 32;

    unsigned int rRows = mRows + fRows - 1;
    unsigned int rCols = mCols + fCols - 1;

    float * data   = (float *)malloc(mRows * mCols * sizeof(float));
    float * filter = (float *)malloc(fRows * fCols * sizeof(float));

    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            data[j + i * mCols] = j * i + 1;

    for (unsigned int i = 0 ; i < fRows ; ++i)
        for (unsigned int j = 0 ; j < fCols ; ++j)
            filter[j + i * fCols] = j * i + 1;

    float * dataPad = (float *)malloc(rRows * rCols * sizeof(float));
    for (unsigned int i = 0 ; i < rRows ; ++i)
        for (unsigned int j = 0 ; j < rCols ; ++j)
            dataPad[j + i * rCols] = 0.0;

    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            dataPad[j + i * rCols] = data[j + i * mCols];

    float * filterPad = (float *)malloc(rRows * rCols * sizeof(float));
    for (unsigned int i = 0 ; i < rRows ; ++i)
        for (unsigned int j = 0 ; j < rCols ; ++j)
            filterPad[j + i * rCols] = 0.0;

    for (unsigned int i = 0 ; i < fRows ; ++i)
        for (unsigned int j = 0 ; j < fCols ; ++j)
            filterPad[j + i * rCols] = filter[j + i * fCols];

    float * result   = (float *)malloc( rRows * rCols * sizeof(float));
    float * solution = (float *)malloc( rRows * rCols * sizeof(float));

    float * d_data;
    check_cuda( cudaMalloc(&d_data, mRows * mCols * sizeof(float)) );

    float * d_filter;
    check_cuda( cudaMalloc(&d_filter, fRows * fCols * sizeof(float)) );

    float * d_result;
    check_cuda( cudaMalloc(&d_result, rRows * rCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_data, data, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

    check_cuda( cudaMemcpy ( d_filter, filter, fRows * fCols * sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::convolution2dMatrixDD<32, 32, float>(d_result, d_data, d_filter, mRows, mCols, fRows, fCols);

    for (unsigned int i = 0 ; i < rRows * rCols ; ++i)
        solution[i] = 0.0;

    convolution2DMatrixCC_CPU(solution, dataPad, filterPad, mRows, mCols, fRows, fCols);

    check_cuda( cudaMemcpy ( result, d_result, rRows * rCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int j = 0 ; j < rRows ; ++j)
        for (unsigned int i = 0 ; i < rCols ; ++i)
            ASSERT_TRUE( (solution[i + j * rCols] - result[i + j * rCols]) / solution[i + j * rCols] < 1e-6 );

    check_cuda( cudaFree(d_data  ) );
    check_cuda( cudaFree(d_filter) );
    check_cuda( cudaFree(d_result) );
    free(data     );
    free(filter   );
    free(result   );
    free(dataPad  );
    free(filterPad);
    free(solution );
}
