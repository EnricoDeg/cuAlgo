/*
 * @file benchmark_convFFT1dCT.cu
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
#include "cuAlgo/API/convFFT1dCT.hpp"

constexpr std::uint32_t next_pow2(std::uint32_t x) {
    if (x <= 1) return 1;

    x--;                    // Important step
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

template<unsigned int Size, unsigned int BlockSize>
void run_benchmark(int batch_size)
{
    constexpr int OutputSize = next_pow2(2 * Size - 1);
    std::cout << "FFTSize: " << Size
              << " BlockSize: " << BlockSize
              <<" Batch size: " << batch_size << std::endl;
    static constexpr unsigned int warmup_size = 5;
    static constexpr unsigned int repeat = 5;

    float * in1 = (float*)malloc(Size * batch_size * sizeof(float));
    float * in2 = (float*)malloc(Size * batch_size * sizeof(float));
    float * out = (float*)malloc(OutputSize * batch_size * sizeof(float));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in1, (float*)in1 + Size * batch_size, gen);
    std::generate((float *)in2, (float*)in2 + Size * batch_size, gen);

    float *d_in1;
    check_cuda( cudaMalloc(&d_in1, Size * batch_size * sizeof(float)) );

    float *d_in2;
    check_cuda( cudaMalloc(&d_in2, Size * batch_size * sizeof(float)) );

    float *d_out;
    check_cuda( cudaMalloc(&d_out, OutputSize * batch_size * sizeof(float)) );

    check_cuda( cudaMemcpy(d_in1 , in1 , Size * batch_size * sizeof(float), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy(d_in2 , in2 , Size * batch_size * sizeof(float), cudaMemcpyHostToDevice ) );

    cuAlgo::conv1dCT_plan<OutputSize / 2>();

    // warmup
    for (unsigned int i = 0; i < warmup_size; ++i)
        cuAlgo::convFFT1dCT<OutputSize, BlockSize>(d_in1, d_in2, d_out, Size, Size, batch_size);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < repeat; ++i)
        cuAlgo::convFFT1dCT<OutputSize, BlockSize>(d_in1, d_in2, d_out, Size, Size, batch_size, 0, false);
    check_cuda( cudaStreamSynchronize(0) );

    check_cuda(cudaEventRecord(stop, 0));
    check_cuda(cudaEventSynchronize(stop));
    float elapsed_mseconds;
    check_cuda(cudaEventElapsedTime(&elapsed_mseconds, start, stop));
    std::cout << "Time taken by function: "
              << "\033[32m" // green
              << elapsed_mseconds * 1000 / repeat
              << "\033[0m"   // reset color
              << " microseconds"
              << std::endl;
    std::cout << "Bytes per second = "
              << float(repeat * Size * batch_size* sizeof(float)) / 1000000000 / (elapsed_mseconds / 1000)
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

    run_benchmark<2048, 1024 >(total_size / 2048); // MixedRadix
    run_benchmark<2048,  512 >(total_size / 2048); // MixedRadix
    run_benchmark<2048,  256 >(total_size / 2048); // MixedRadix
    run_benchmark<2048,  128 >(total_size / 2048); // MixedRadix
    run_benchmark<2048,   64 >(total_size / 2048); // MixedRadix

    run_benchmark<4096, 1024 >(total_size / 4096); // MixedRadix
    run_benchmark<4096,  512 >(total_size / 4096); // MixedRadix
    run_benchmark<4096,  256 >(total_size / 4096); // MixedRadix
    run_benchmark<4096,  128 >(total_size / 4096); // MixedRadix
    run_benchmark<4096,   64 >(total_size / 4096); // MixedRadix

    run_benchmark<8192, 1024 >(total_size / 8192); // MixedRadix
    run_benchmark<8192,  512 >(total_size / 8192); // MixedRadix
    run_benchmark<8192,  256 >(total_size / 8192); // MixedRadix
    run_benchmark<8192,  128 >(total_size / 8192); // MixedRadix
    run_benchmark<8192,   64 >(total_size / 8192); // MixedRadix

    return 0;
}
