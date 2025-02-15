/*
 * @file test_histogram.cu
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
#include "cuAlgo/API/histogram.hpp"

TEST(histogram, basic_unsigned_char) {

    unsigned int size = 1024 * 1024;
    constexpr unsigned int bin_size = 256;

#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
    cudaDeviceProp prop;
    cudaGetDeviceProperties ( &prop, 0 );
    std::cout << "device cc: " << prop.major << "," << prop.minor << std::endl;
#endif
    unsigned char * data = (unsigned char *)malloc(size * sizeof(unsigned char));
    for (unsigned int i = 0 ; i < size ; ++i)
        data[i] = (unsigned char)(i % 256);

    unsigned int * histo = (unsigned int *)malloc(bin_size * sizeof(unsigned int));
    unsigned int * solution = (unsigned int *)malloc(bin_size * sizeof(unsigned int));

    unsigned char *d_data;
    check_cuda( cudaMalloc(&d_data, size * sizeof(unsigned char)) );

    unsigned int *d_histo;
    check_cuda( cudaMalloc(&d_histo, bin_size * sizeof(unsigned int)) );

    check_cuda( cudaMemcpy ( d_data, data, size * sizeof(unsigned char), cudaMemcpyHostToDevice ) );

    cuAlgo::histogram<1024, unsigned char, bin_size>(d_data, size, d_histo);

    for (unsigned int i = 0 ; i < bin_size ; ++i)
        solution[i] = 0;

    for (unsigned int i = 0 ; i < size ; ++i)
        solution[ data[i] ]++;

    check_cuda( cudaMemcpy ( histo, d_histo, bin_size * sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0 ; i < bin_size ; ++i)
        ASSERT_EQ( solution[i], histo[i] );

    check_cuda( cudaFree(d_data) );
    check_cuda( cudaFree(d_histo) );
    free(data);
    free(histo);
    free(solution);
}
