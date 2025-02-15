/*
 * @file test_dshear1dMatrix.cu
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
#include "cuAlgo/config/dshear1dMatrix_config.hpp"
#include "cuAlgo/API/dshear1dMatrix.hpp"

void dshear1dMatrix_CPU(float * idata, float *odata,
                        long int k, unsigned int dim,
                        unsigned int mRows, unsigned int mCols) {

    if ( dim == 0 ) {

        for (unsigned int j = 0; j < mCols; ++j) {

            long int shift = -k*(mCols / 2 - j);
            if (shift < 0) {

                for (unsigned int i = 0; i < mRows+shift; ++i )
                    odata[i * mCols + j] = idata[(i-shift) * mCols + j];
                for (unsigned int i = mRows+shift; i < mRows; ++i)
                    odata[i * mCols + j] = idata[(i - (mRows+shift)) * mCols + j];
            } else {

                for (unsigned int i = 0; i < shift; ++i)
                    odata[i * mCols + j] = idata[(mRows-shift+i) * mCols + j];
                for (unsigned int i = shift; i < mRows; ++i)
                    odata[i * mCols + j] = idata[(i-shift) * mCols + j];
            }
        }

    } else if ( dim == 1 ) {

        for (unsigned int i = 0; i < mRows; ++i) {

            long int shift = -k*(mRows / 2 - i);
            if (shift < 0) {
                for (unsigned int j = 0; j < mCols+shift; ++j )
                    odata[i * mCols + j] = idata[i * mCols + (j-shift)];
                for (unsigned int j = mCols+shift; j < mCols; ++j)
                    odata[i * mCols + j] = idata[i * mCols + (j - (mCols + shift))];
            } else {
                for (unsigned int j = 0; j < shift; ++j)
                    odata[i * mCols + j] = idata[i * mCols + (mCols-shift+j)];
                for (unsigned int j = shift; j < mCols; ++j)
                    odata[i * mCols + j] = idata[i * mCols + (j-shift)];
            }
        }
    }
}

TEST(dshear1dMatrix, default_values_dim0) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1024;

    float * idata = (float *)malloc(mRows * mCols * sizeof(float));
    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            idata[j + i * mCols] = j * i + 1;

    float * odata    = (float *)malloc(mRows * mCols * sizeof(float));
    float * solution = (float *)malloc(mRows * mCols * sizeof(float));

    float *d_idata;
    check_cuda( cudaMalloc(&d_idata, mRows * mCols * sizeof(float)) );

    float *d_odata;
    check_cuda( cudaMalloc(&d_odata, mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::dshear1dMatrix<float>(d_idata,
                                  d_odata,
                                  1,
                                  0,
                                  mRows,
                                  mCols,
                                  cuAlgo::getConfigParams_dshear1dMatrix<float>());

    for (unsigned int i = 0 ; i < mRows * mCols ; ++i)
        solution[i] = 0;

    dshear1dMatrix_CPU(idata, solution, 1, 0, mRows, mCols);

    check_cuda( cudaMemcpy ( odata, d_odata, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int j = 0 ; j < mRows ; ++j)
        for (unsigned int i = 0 ; i < mCols ; ++i)
            ASSERT_TRUE( (solution[i + j * mCols] - odata[i + j * mCols]) / solution[i + j * mCols] < 1e-6 );

    check_cuda( cudaFree(d_idata) );
    check_cuda( cudaFree(d_odata) );
    free(idata);
    free(odata);
    free(solution);
}

TEST(dshear1dMatrix, default_values_dim1) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1024;

    float * idata = (float *)malloc(mRows * mCols * sizeof(float));
    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            idata[j + i * mCols] = j * i + 1;

    float * odata    = (float *)malloc(mRows * mCols * sizeof(float));
    float * solution = (float *)malloc(mRows * mCols * sizeof(float));

    float *d_idata;
    check_cuda( cudaMalloc(&d_idata, mRows * mCols * sizeof(float)) );

    float *d_odata;
    check_cuda( cudaMalloc(&d_odata, mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::dshear1dMatrix<float>(d_idata,
                                  d_odata,
                                  1,
                                  1,
                                  mRows,
                                  mCols);

    for (unsigned int i = 0 ; i < mRows * mCols ; ++i)
        solution[i] = 0;

    dshear1dMatrix_CPU(idata, solution, 1, 1, mRows, mCols);

    check_cuda( cudaMemcpy ( odata, d_odata, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int j = 0 ; j < mRows ; ++j)
        for (unsigned int i = 0 ; i < mCols ; ++i)
            ASSERT_TRUE( (solution[i + j * mCols] - odata[i + j * mCols]) / solution[i + j * mCols] < 1e-6 );

    check_cuda( cudaFree(d_idata) );
    check_cuda( cudaFree(d_odata) );
    free(idata);
    free(odata);
    free(solution);
}

TEST(dshear1dMatrix, config) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1024;

    float * idata = (float *)malloc(mRows * mCols * sizeof(float));
    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            idata[j + i * mCols] = j * i + 1;

    float * odata    = (float *)malloc(mRows * mCols * sizeof(float));
    float * solution = (float *)malloc(mRows * mCols * sizeof(float));

    float *d_idata;
    check_cuda( cudaMalloc(&d_idata, mRows * mCols * sizeof(float)) );

    float *d_odata;
    check_cuda( cudaMalloc(&d_odata, mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

    using config = dshear_config<32, 32, 1>;
    const dshear_config_params params = cuAlgo::getConfigParams_dshear1dMatrix<float, config>();
    cuAlgo::dshear1dMatrix<float, config>(d_idata, d_odata, 1, 1, mRows, mCols, params);

    for (unsigned int i = 0 ; i < mRows * mCols ; ++i)
        solution[i] = 0;

    dshear1dMatrix_CPU(idata, solution, 1, 1, mRows, mCols);

    check_cuda( cudaMemcpy ( odata, d_odata, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int j = 0 ; j < mRows ; ++j)
        for (unsigned int i = 0 ; i < mCols ; ++i)
            ASSERT_TRUE( (solution[i + j * mCols] - odata[i + j * mCols]) / solution[i + j * mCols] < 1e-6 );

    check_cuda( cudaFree(d_idata) );
    check_cuda( cudaFree(d_odata) );
    free(idata);
    free(odata);
    free(solution);
}
