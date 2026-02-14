/*
 * @file benchmark_fft1dBailey.cu
 *
 * @copyright Copyright (C) 2026 Enrico Degregori <enrico.degregori@gmail.com>
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
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "cuAlgo/API/fft1dBailey.hpp"

template<unsigned int Size1, unsigned int Size2, unsigned int BlockSize>
void run_benchmark(int batch_size)
{
    constexpr unsigned int Size = Size1 * Size2;
    std::cout << "FFTSize: " << Size
              << " BlockSize: " << BlockSize
              <<" Batch size: " << batch_size << std::endl;
    static constexpr unsigned int warmup_size = 5;
    static constexpr unsigned int repeat = 5;

    float2 * in  = (float2*)malloc(Size * batch_size * sizeof(float2));
    float2 * out = (float2*)malloc(Size * batch_size * sizeof(float2));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in, (float*)in + 2 * Size * batch_size, gen);

    float2 *d_in;
    check_cuda( cudaMalloc(&d_in, Size * batch_size * sizeof(float2)) );

    float2 *d_out;
    check_cuda( cudaMalloc(&d_out, Size * batch_size * sizeof(float2)) );

    check_cuda( cudaMemcpy(d_in , in , Size * batch_size * sizeof(float2), cudaMemcpyHostToDevice ) );

    cuAlgo::fft1dCT_plan<Size2>();
    // warmup
    for (unsigned int i = 0; i < warmup_size; ++i)
        cuAlgo::fft1dBailey<Size1, Size2, BlockSize>(d_in, d_out, batch_size);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < repeat; ++i)
        cuAlgo::fft1dBailey<Size1, Size2, BlockSize>(d_in, d_out, batch_size, 0, true);
    check_cuda( cudaStreamSynchronize(0) );

    check_cuda(cudaEventRecord(stop, 0));
    check_cuda(cudaEventSynchronize(stop));
    float elapsed_mseconds;
    check_cuda(cudaEventElapsedTime(&elapsed_mseconds, start, stop));
    std::cout << "Time taken by function: "
              << elapsed_mseconds * 1000 / repeat
              << " microseconds"
              << std::endl;
    std::cout << "Bytes per second = "
              << float(repeat * Size * batch_size* sizeof(float2)) / 1000000000 / (elapsed_mseconds / 1000)
              << " Gb / s"
              << std::endl;

    // Destroy CUDA events
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    check_cuda( cudaFree(d_in) );
    check_cuda( cudaFree(d_out) );
    free(in);
    free(out);
}

int main() {
    static constexpr unsigned int total_size = 1024 * 8192;

    // run_benchmark<32, 32,  64>(total_size / 1024); // WarpFFT
    // run_benchmark<32, 32, 128>(total_size / 1024); // WarpFFT
    // run_benchmark<32, 32, 256>(total_size / 1024); // WarpFFT
    // run_benchmark<32, 32, 512>(total_size / 1024); // WarpFFT

    // run_benchmark<2, 2048 / 2, 256>(total_size / 2048); // Radix2
    // run_benchmark<4, 2048 / 4, 128>(total_size / 2048); // Radix4

    // run_benchmark<2, 4096 / 2, 256>(total_size / 4096); // Radix2
    run_benchmark<4, 4096 / 4, 256>(total_size / 4096); // Radix4
    // run_benchmark<4, 4096 / 4, 128>(total_size / 4096); // Radix4
    // run_benchmark<4, 4096 / 4,  64>(total_size / 4096); // Radix4
    // run_benchmark<8, 4096 / 8, 128>(total_size / 4096); // Radix4

    return 0;
}
