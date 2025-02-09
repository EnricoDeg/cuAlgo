/*
 * @file test_fliplr1dMatrix.cu
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
#include "src/API/fliplr1dMatrix.hpp"
#include <gtest/gtest.h>

using namespace std::chrono;

void fliplrMatrix_CPU(float *idata, float *odata, unsigned int dim,
                      unsigned int mRows, unsigned int mCols) {

    if ( dim == 0 ) {

        for (unsigned int i = 0; i < mRows; ++i) {

            const float * __restrict__ in = idata + (mRows - 1 - i) * mCols;
            float * __restrict__ out = odata + i * mCols;
            for (unsigned int j = 0; j < mCols; ++j)
                out[j] = in[j];
        }
    } else if (dim == 1) {

        for (unsigned int i = 0; i < mRows; ++i) {

            const float * __restrict__ in = idata + i*mCols;
            float * __restrict__ out = odata + i*mCols;
            for (unsigned int j = 0; j < mCols; ++j)
                out[j] = in[mCols - 1 -j];
        }
    }
}

TEST(fliplr1dMatrix, default_dim0_even) {

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

    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 0, mRows, mCols);

    fliplrMatrix_CPU(input, solution, 0, mRows, mCols);

    check_cuda( cudaMemcpy ( input, d_input, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            ASSERT_EQ( input[j + i * mCols] , solution[j + i * mCols] );

    check_cuda( cudaFree(d_input) );
    free(input);
    free(solution);
}

TEST(fliplr1dMatrix, default_dim0_odd) {

    unsigned int mRows = 1025;
    unsigned int mCols = 1024;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));
    float * solution = (float *)malloc(mRows * mCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 0, mRows, mCols);

    fliplrMatrix_CPU(input, solution, 0, mRows, mCols);

    check_cuda( cudaMemcpy ( input, d_input, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            ASSERT_EQ( input[j + i * mCols] , solution[j + i * mCols] );

    check_cuda( cudaFree(d_input) );
    free(input);
    free(solution);
}

TEST(fliplr1dMatrix, default_dim1_even) {

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

    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 1, mRows, mCols);

    fliplrMatrix_CPU(input, solution, 1, mRows, mCols);

    check_cuda( cudaMemcpy ( input, d_input, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            ASSERT_EQ( input[j + i * mCols] , solution[j + i * mCols] );

    check_cuda( cudaFree(d_input) );
    free(input);
    free(solution);
}

TEST(fliplr1dMatrix, default_dim1_odd) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1025;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));
    float * solution = (float *)malloc(mRows * mCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 1, mRows, mCols);

    fliplrMatrix_CPU(input, solution, 1, mRows, mCols);

    check_cuda( cudaMemcpy ( input, d_input, mRows * mCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            ASSERT_EQ( input[j + i * mCols] , solution[j + i * mCols] );

    check_cuda( cudaFree(d_input) );
    free(input);
    free(solution);
}

TEST(fliplr1dMatrix, performance_dim0_even) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1024;
    unsigned int iterations = 10;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    // warm-up
    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 0, mRows, mCols);

    auto start = high_resolution_clock::now();
    for (unsigned int i = 0; i < iterations; ++i)
        cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 0, mRows, mCols);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / iterations << std::endl;
    ASSERT_TRUE(duration.count() / iterations < 20);

    check_cuda( cudaFree(d_input) );
    free(input);
}

TEST(fliplr1dMatrix, performance_dim1_even) {

    unsigned int mRows = 1024;
    unsigned int mCols = 1024;
    unsigned int iterations = 10;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    // warm-up
    cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 1, mRows, mCols);

    auto start = high_resolution_clock::now();
    for (unsigned int i = 0; i < iterations; ++i)
        cuAlgo::fliplr1dMatrix<32, 32, float>(d_input, 1, mRows, mCols);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / iterations << std::endl;
    ASSERT_TRUE(duration.count() / iterations < 20);

    check_cuda( cudaFree(d_input) );
    free(input);
}
