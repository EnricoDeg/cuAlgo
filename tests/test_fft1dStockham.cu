/*
 * @file test_fft1dStockham.cu
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
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <gtest/gtest.h>
#include "cuAlgo/API/fft1dStockham.hpp"

inline float2 twiddle_cpu(int k, int m) {
    float angle = -2.0 * M_PI * k / m;
    return { std::cos(angle), std::sin(angle) };
}

void stockham_fft(float2 * x, unsigned int N)
{
    assert((N & (N - 1)) == 0); // power of 2

    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

    std::vector<float2> y(N);

    float2* in  = x;
    float2* out = y.data();

    for (int s = 0; s < log2N; ++s) {
        int m  = 1 << (s + 1);   // FFT size this stage
        int mh = m >> 1;

        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < mh; ++j) {

                // ---------
                // INPUT PERMUTATION (this is the Stockham part)
                // ---------
                int i0 = k / 2 + j;
                int i1 = i0 + N / 2;

                float2 a = in[i0];
                float2 b = in[i1];

                float2 w = twiddle_cpu(j, m);
                float2 t = cmul(b, w);

                // ---------
                // OUTPUT IS WRITTEN LINEARLY
                // ---------
                out[k + j]      = cadd(a, t);
                out[k + j + mh] = csub(a, t);
            }
        }

        std::swap(in, out);
    }

    // Final result may be in temp buffer
    if (in != x) {
        for (int i = 0; i < N; ++i)
            x[i] = in[i];
    }
}

template<unsigned int Size>
void run_single_test()
{
    float2 * in  = (float2*)malloc(Size * sizeof(float2));
    float2 * out = (float2*)malloc(Size * sizeof(float2));
    float2 * solution = (float2 *)malloc(Size * sizeof(float2));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in, (float*)in + 2 * Size, gen);

    float2 *d_in;
    check_cuda( cudaMalloc(&d_in, Size * sizeof(float2)) );

    float2 *d_out;
    check_cuda( cudaMalloc(&d_out, Size * sizeof(float2)) );

    check_cuda( cudaMemcpy(d_in , in , Size * sizeof(float2), cudaMemcpyHostToDevice ) );

    for(int i = 0; i < Size; ++i)
    {
        solution[i] = in[i];
    }

    // CPU reference
    stockham_fft(solution, Size);

    // GPU
    cuAlgo::fft1dStockham<Size>(d_in, d_out);

    check_cuda( cudaMemcpy ( out, d_out, Size * sizeof(float2), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < Size; ++i)
    {
        if(solution[i].x > 1e-5 && out[i].x > 1e-5)
            ASSERT_TRUE(std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x) < 1e-3);
        if(solution[i].y > 1e-5 && out[i].y > 1e-5)
            ASSERT_TRUE(std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y) < 1e-3);
    }

    check_cuda( cudaFree(d_in) );
    check_cuda( cudaFree(d_out) );
    free(in);
    free(out);
    free(solution);
}

TEST(CT_Stockham, size_1024) {

    constexpr unsigned int size = 1024;
    run_single_test<size>();
}

TEST(CT_Stockham, size_512) {

    constexpr unsigned int size = 512;
    run_single_test<size>();
}
