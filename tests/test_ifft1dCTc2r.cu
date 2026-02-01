/*
 * @file test_ifft1dCTr2c.cu
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
#include <gtest/gtest.h>
#include "cuAlgo/API/fft1dCTr2c.hpp"
#include "cuAlgo/API/ifft1dCTc2r.hpp"

void bit_reverse(float2 * a, unsigned int N)
{
    const unsigned int logN = std::log2(N);

    for (unsigned int i = 0; i < N; ++i) {
        unsigned int x = i, j = 0;
        for (unsigned int b = 0; b < logN; ++b) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i)
            std::swap(a[i], a[j]);
    }
}

void fft_cpu(float2* a, unsigned int N)
{
    assert((N & (N - 1)) == 0);

    bit_reverse(a, N);

    for (unsigned int len = 2; len <= N; len <<= 1) {
        unsigned int half = len >> 1;
        float ang = -2.0f * M_PI / len;

        for (unsigned int i = 0; i < N; i += len) {
            for (unsigned int k = 0; k < half; ++k) {
                float theta = ang * k;
                float2 w = {std::cos(theta), std::sin(theta)};

                float2 u = a[i + k];
                float2 v = cmul(w, a[i + k + half]);

                a[i + k]        = cadd(u, v);
                a[i + k + half] = csub(u, v);
            }
        }
    }
}

template<unsigned int Size, unsigned int BlockSize>
void run_single_test()
{
    float * in        = (float*)malloc(Size * sizeof(float));
    float * out       = (float*)malloc((Size + 2) * sizeof(float));
    float * out2      = (float*)malloc(Size * sizeof(float));
    float2 * solution = (float2 *)malloc(Size * sizeof(float2));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in, (float*)in + Size, gen);

    float *d_in;
    check_cuda( cudaMalloc(&d_in, Size * sizeof(float)) );

    float *d_out;
    check_cuda( cudaMalloc(&d_out, (Size + 2) * sizeof(float)) );

    float *d_out2;
    check_cuda( cudaMalloc(&d_out2, Size * sizeof(float)) );

    check_cuda( cudaMemcpy(d_in , in , Size * sizeof(float), cudaMemcpyHostToDevice ) );

    for(int i = 0; i < Size; ++i)
    {
        solution[i].x = in[i];
        solution[i].y = 0.0f;
    }

    // CPU reference
    fft_cpu(solution, Size);

    // GPU
    cuAlgo::fft1dCT_plan<Size / 2>();
    cuAlgo::ifft1dCT_plan<Size / 2>();
    cuAlgo::fft1dCTr2c<Size, BlockSize>(d_in, d_out, 1);
    cuAlgo::ifft1dCTc2r<Size, BlockSize>(d_out, d_out2, 1);

    check_cuda( cudaMemcpy ( out, d_out, (Size + 2) * sizeof(float), cudaMemcpyDeviceToHost ) );
    check_cuda( cudaMemcpy ( out2, d_out2, (Size) * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < Size; ++i)
    {
        // std::cout << i << ": " << in[i] << " --- " << out2[i] << std::endl;
        ASSERT_TRUE(std::abs(in[i] - out2[i]) / std::abs(in[i]) < 1e-3);
    }

    check_cuda( cudaFree(d_in) );
    free(in);
    free(solution);
}

TEST(CT_IFFTc2r, size_256) {
    // Radix4
    constexpr unsigned int size = 256;
    run_single_test<size, 64>();
}

TEST(CT_IFFTc2r, size_512) {
    // MixedRadix
    constexpr unsigned int size = 512;
    run_single_test<size, 64>();
}
