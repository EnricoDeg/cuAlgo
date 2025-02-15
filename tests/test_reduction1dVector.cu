/*
 * @file test_reduction1dVector.cu
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
#include "cuAlgo/API/reduction1dVector.hpp"

TEST(reduction1dVector, default_value) {

    unsigned int nblocks = 4096;
    unsigned int size = 1024*nblocks;
    int * input = (int *)malloc(size * sizeof(int));
    int * output = (int *)malloc(sizeof(int));
    int * solution = (int *)malloc(sizeof(int));
    for(unsigned int i = 0; i < nblocks; ++i)
        for (unsigned int j = 0; j < 1024 ; ++j)
            input[j + i*1024] = j;

    output[0] = 0;

    int *d_input;
    check_cuda( cudaMalloc(&d_input, size*sizeof(int)) );

    int *d_output;
    check_cuda( cudaMalloc(&d_output, sizeof(int)) );

    check_cuda( cudaMemcpy ( d_input, input, (unsigned int)size*sizeof(int), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy ( d_output, output, sizeof(int), cudaMemcpyHostToDevice ) );

    cuAlgo::reduction1dVector2<int, 1024, 2>(d_input, d_output, size);

    solution[0] = 0;
    for(unsigned int i = 0; i < size; ++i)
        solution[0] += input[i];

    check_cuda( cudaMemcpy ( output, d_output, sizeof(int), cudaMemcpyDeviceToHost ) );

    ASSERT_EQ(solution[0], output[0]);

    check_cuda( cudaFree(d_input ) );
    check_cuda( cudaFree(d_output) );
    free(input);
    free(output);
    free(solution);
}
