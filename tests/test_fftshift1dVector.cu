/*
 * @file test_fftshift1dVector.cu
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
#include "cuAlgo/API/fftshift1dVector.hpp"

#define BLOCKSIZE 1024

void fftshiftVectorCPU(int * idata, int * odata, unsigned int size) {

    int * __restrict in  = idata ;
    int * __restrict out = odata ;

    if (size % 2 == 0) {

        for (unsigned int j = 0; j < size / 2; ++j) {
            out[j] = in[size / 2 + j];
        }
        for (unsigned int j = size / 2; j < size; ++j) {
            out[j] = in[j - size / 2];
        }
    } else {

        for (unsigned int j = 0; j < ( size - 1 ) / 2; ++j) {
            out[j] = in[(size + 1 )/ 2 + j];
        }
        for (unsigned int j = ( size - 1 ) / 2; j < size; ++j) {
            out[j] = in[j - ( size - 1 ) / 2];
        }
    }
}

TEST(fftshiftVector, default_even) {

    unsigned int nblocks = 2;
    unsigned int size = BLOCKSIZE * nblocks;

    int * input    = (int *)malloc(size * sizeof(int));
    int * output   = (int *)malloc(size * sizeof(int));
    int * solution = (int *)malloc(size * sizeof(int));

    for(unsigned int i = 0; i < nblocks; ++i)
        for (unsigned int j = 0; j < BLOCKSIZE ; ++j)
            input[j + i*BLOCKSIZE] = j;

    int *d_input;
    check_cuda( cudaMalloc(&d_input , size * sizeof(int)) );

    int *d_output;
    check_cuda( cudaMalloc(&d_output, size * sizeof(int)) );

    check_cuda( cudaMemcpy ( d_input, input, (unsigned int)size*sizeof(int), cudaMemcpyHostToDevice ) );

    cuAlgo::fftshift1dVector<BLOCKSIZE, int>(d_input, d_output, size);

    fftshiftVectorCPU(input, solution, size);

    check_cuda( cudaMemcpy ( output, d_output, (unsigned int)size*sizeof(int), cudaMemcpyDeviceToHost ) );

    for(unsigned int i = 0; i < nblocks; ++i)
        for (unsigned int j = 0; j < BLOCKSIZE ; ++j)
            ASSERT_EQ( output[j + i * BLOCKSIZE] , solution[j + i * BLOCKSIZE] );

    check_cuda( cudaFree(d_input) );
    check_cuda( cudaFree(d_output) );
    free(input);
    free(output);
    free(solution);
}
