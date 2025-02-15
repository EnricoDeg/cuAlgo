/*
 * @file test_padarray2dMatrix.cu
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
#include "thrust/complex.h"
#include <gtest/gtest.h>
#include "cuAlgo/API/padarray2dMatrix.hpp"

using namespace std::chrono;

template <class T>
void padarray_CPU(T * idata, T * odata,
                  unsigned int nrows, unsigned int ncols,
                  unsigned int mRows, unsigned int mCols) {

    unsigned int offsetRows = ( nrows - mRows ) / 2 + ( nrows - mRows ) % 2;
    unsigned int offsetCols = ( ncols - mCols ) / 2 + ( ncols - mCols ) % 2;
    for (size_t i = offsetRows; i < offsetRows + mRows; ++i) {
        T * __restrict in  = idata  + (i - offsetRows) * mCols ;
        T * __restrict out = odata + i*ncols;
        memcpy(out + offsetCols, in, mCols * sizeof(T));
    }
}

TEST(padarray2dMatrix, default_float) {

    unsigned int nRows = 1024;
    unsigned int nCols = 1024;
    unsigned int mRows =  512;
    unsigned int mCols =  512;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));
    float * output   = (float *)malloc(nRows * nCols * sizeof(float));
    float * solution = (float *)malloc(nRows * nCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    float *d_output;
    check_cuda( cudaMalloc(&d_output , nRows * nCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::padarray2dMatrix<32, 32, float>(d_input, d_output, nRows, nCols, mRows, mCols);

    padarray_CPU<float>(input, solution, nRows, nCols, mRows, mCols);

    check_cuda( cudaMemcpy ( output, d_output, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < nRows; ++i)
        for (unsigned int j = 0; j < nCols ; ++j)
            ASSERT_EQ( output[j + i * nCols] , solution[j + i * nCols] );

    check_cuda( cudaFree(d_input ) );
    check_cuda( cudaFree(d_output) );
    free(input   );
    free(output  );
    free(solution);
}

TEST(padarray2dMatrix, default_complex_float) {

    unsigned int nRows = 1024;
    unsigned int nCols = 1024;
    unsigned int mRows =  512;
    unsigned int mCols =  512;

    thrust::complex<float> * input    = (thrust::complex<float> *)malloc(mRows * mCols * sizeof(thrust::complex<float>));
    thrust::complex<float> * output   = (thrust::complex<float> *)malloc(nRows * nCols * sizeof(thrust::complex<float>));
    thrust::complex<float> * solution = (thrust::complex<float> *)malloc(nRows * nCols * sizeof(thrust::complex<float>));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = thrust::complex<float>((float)(i) * j + 1, (float)(i) + j + 1);

    thrust::complex<float> *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(thrust::complex<float>)) );

    thrust::complex<float> *d_output;
    check_cuda( cudaMalloc(&d_output , nRows * nCols * sizeof(thrust::complex<float>)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(thrust::complex<float>), cudaMemcpyHostToDevice ) );

    cuAlgo::padarray2dMatrix<32, 32, thrust::complex<float>>(d_input, d_output, nRows, nCols, mRows, mCols);

    padarray_CPU<thrust::complex<float>>(input, solution, nRows, nCols, mRows, mCols);

    check_cuda( cudaMemcpy ( output, d_output, nRows * nCols * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < nRows; ++i)
        for (unsigned int j = 0; j < nCols ; ++j) {
            ASSERT_EQ( output[j + i * nCols].real() , solution[j + i * nCols].real() );
            ASSERT_EQ( output[j + i * nCols].imag() , solution[j + i * nCols].imag() );
        }

    check_cuda( cudaFree(d_input ) );
    check_cuda( cudaFree(d_output) );
    free(input   );
    free(output  );
    free(solution);
}

TEST(padarray2dMatrix, performance) {

    unsigned int nRows      = 1024;
    unsigned int nCols      = 1024;
    unsigned int mRows      =  512;
    unsigned int mCols      =  512;
    unsigned int iterations =   10;

    float * input    = (float *)malloc(mRows * mCols * sizeof(float));

    for(unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols ; ++j)
            input[j + i*mCols] = i * j + 1;

    float *d_input;
    check_cuda( cudaMalloc(&d_input , mRows * mCols * sizeof(float)) );

    float *d_output;
    check_cuda( cudaMalloc(&d_output , nRows * nCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_input, input, mRows * mCols *sizeof(float), cudaMemcpyHostToDevice ) );

    // warm-up
    cuAlgo::padarray2dMatrix<32, 32, float>(d_input, d_output, nRows, nCols, mRows, mCols);

    auto start = high_resolution_clock::now();
    for (unsigned int i = 0; i < iterations; ++i)
        cuAlgo::padarray2dMatrix<32, 32, float>(d_input, d_output, nRows, nCols, mRows, mCols);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / iterations << std::endl;
    ASSERT_TRUE(duration.count() / iterations < 25);

    check_cuda( cudaFree(d_input ) );
    check_cuda( cudaFree(d_output) );
    free(input   );
}
