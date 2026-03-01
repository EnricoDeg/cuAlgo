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
#include "cuAlgo/API/convFFT1dBailey.hpp"

constexpr std::uint32_t next_pow2(std::uint32_t x)
{
    if (x <= 1) return 1;

    x--;                    // Important step
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

template<
unsigned int Size1,
unsigned int IDataSize,
unsigned int BlockSize>
void run_benchmark(int batch_size)
{
    constexpr int OutputSize = next_pow2(2 * IDataSize - 1);
    constexpr int Size2 = OutputSize / Size1;
    static_assert(Size2 == 1024);

    std::cout << "FFTSize: " << OutputSize
              << " BlockSize: " << BlockSize
              <<" Batch size: " << batch_size << std::endl;

    static constexpr unsigned int warmup_size = 5;
    static constexpr unsigned int repeat = 5;

    float2 * in1 = (float2*)malloc(IDataSize * batch_size * sizeof(float2));
    float2 * in2 = (float2*)malloc(IDataSize * batch_size * sizeof(float2));
    float2 * out = (float2*)malloc(OutputSize * batch_size * sizeof(float2));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in1, (float*)in1 + 2 * IDataSize * batch_size, gen);
    std::generate((float *)in2, (float*)in2 + 2 * IDataSize * batch_size, gen);

    float2 *d_in1;
    check_cuda( cudaMalloc(&d_in1, IDataSize * batch_size * sizeof(float2)) );

    float2 *d_in2;
    check_cuda( cudaMalloc(&d_in2, IDataSize * batch_size * sizeof(float2)) );

    float2 *d_out;
    check_cuda( cudaMalloc(&d_out, OutputSize * batch_size * sizeof(float2)) );

    check_cuda( cudaMemcpy(d_in1 , in1 , IDataSize * batch_size * sizeof(float2), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy(d_in2 , in2 , IDataSize * batch_size * sizeof(float2), cudaMemcpyHostToDevice ) );

    // cuAlgo::fft1dCT_plan<Size2>();
    // warmup
    for (unsigned int i = 0; i < warmup_size; ++i)
        cuAlgo::convFFT1dBailey<Size1, Size2, IDataSize, IDataSize, BlockSize>(
            d_in1, d_in2, d_out, batch_size);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < repeat; ++i)
        cuAlgo::convFFT1dBailey<Size1, Size2, IDataSize, IDataSize, BlockSize>(
            d_in1, d_in2, d_out, batch_size, 0, true);
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
              << float(repeat * OutputSize * batch_size* sizeof(float2)) / 1000000000 / (elapsed_mseconds / 1000)
              << " Gb / s"
              << std::endl;

    // Destroy CUDA events
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    check_cuda( cudaFree(d_in1) );
    check_cuda( cudaFree(d_in2) );
    check_cuda( cudaFree(d_out) );
    free(in1);
    free(in2);
    free(out);
}

int main() {
    static constexpr unsigned int total_size = 1024 * 8192;

    run_benchmark<4, 2048, 256>(total_size / 4096); // Radix4

    return 0;
}
