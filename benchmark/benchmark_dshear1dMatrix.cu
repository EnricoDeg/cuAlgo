/*
 * @file benchmark_dshear1dMatrix.cu
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
#include "API/dshear1dMatrix.hpp"
#include "config/dshear1dMatrix_config.hpp"

template<
typename T,
unsigned int threadsPerBlockX,
unsigned int threadsPerBlockY,
unsigned int itemsPerThread
>
void run_benchmark() {

    unsigned int mRows = 4096;
    unsigned int mCols = 4096;

    float * idata = (float *)malloc(mRows * mCols * sizeof(float));
    for (unsigned int i = 0 ; i < mRows ; ++i)
        for (unsigned int j = 0 ; j < mCols ; ++j)
            idata[j + i * mCols] = j * i + 1;

    float *d_idata;
    check_cuda( cudaMalloc(&d_idata, mRows * mCols * sizeof(float)) );

    float *d_odata;
    check_cuda( cudaMalloc(&d_odata, mRows * mCols * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_idata, idata, mRows * mCols * sizeof(float), cudaMemcpyHostToDevice ) );

    for (unsigned int i = 0; i < 5; ++i)
        cuAlgo::dshear1dMatrix<T, dshear_config<threadsPerBlockX, threadsPerBlockY, itemsPerThread>>(d_idata, d_odata, 1, 1, mRows, mCols);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < 10; ++i)
        cuAlgo::dshear1dMatrix<T, dshear_config<threadsPerBlockX, threadsPerBlockY, itemsPerThread>>(d_idata, d_odata, 1, 1, mRows, mCols, 0);

    check_cuda( cudaStreamSynchronize(0) );

    check_cuda(cudaEventRecord(stop, 0));
    check_cuda(cudaEventSynchronize(stop));
    float elapsed_mseconds;
    check_cuda(cudaEventElapsedTime(&elapsed_mseconds, start, stop));
    std::cout << "Time taken by function: " << elapsed_mseconds * 1000 / 10 << " microseconds" << std::endl;
    // std::cout << "Bytes per second = " << float(mRows * mCols * sizeof(T)) / 1000000000 / (float(duration.count()) / 1000000) << " Gb / s" << std::endl;

    // Destroy CUDA events
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    check_cuda( cudaFree(d_idata) );
    check_cuda( cudaFree(d_odata) );
    free(idata);
}

int main() {

    run_benchmark<float, 32, 32, 1>();
    run_benchmark<float, 32, 32, 2>();
    run_benchmark<float, 32, 32, 4>();
}