/*
 * @file test_convFFT1dCT.cu
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
#include "cuAlgo/API/convFFT1dCT.hpp"

using std::vector;
using std::swap;

// Bit reversal permutation
void bitReverse(vector<float2> &a) {
    int n = a.size();
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
}

// Iterative FFT
void fft(vector<float2> &a, bool invert) {
    int n = a.size();
    bitReverse(a);

    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * M_PI / len * (invert ? -1 : 1);

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len/2; j++) {

                float theta = ang * j;
                float2 w = make(std::cos(theta), std::sin(theta));

                float2 u = a[i + j];
                float2 v = cmul(w, a[i + j + len/2]);

                a[i + j] = cadd(u, v);
                a[i + j + len/2] = csub(u, v);
                // w = cmul(w, wlen);
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i].x /= n;
            a[i].y /= n;
        }
    }
}

// Convolution using FFT
vector<float> convolution(const float * a, const float * b, int N) {
    int n = 1;
    while (n < N + N) n <<= 1;

    vector<float2> fa(n), fb(n);
    for (size_t i = 0; i < N; i++) fa[i] = make(a[i], 0);
    for (size_t i = 0; i < N; i++) fb[i] = make(b[i], 0);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++) fa[i] = cmul(fa[i], fb[i]);
    fft(fa, true);

    vector<float> result(n);
    for (size_t i = 0; i < result.size(); i++) result[i] = fa[i].x;
    return result;
}

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
void run_single_test()
{
    constexpr int OutputSize = next_pow2(2 * Size - 1);

    float * in1      = (float*)malloc(Size * sizeof(float));
    float * in2      = (float*)malloc(Size * sizeof(float));
    float * out      = (float*)malloc(OutputSize * sizeof(float));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in1, (float*)in1 + Size, gen);
    std::generate((float *)in2, (float*)in2 + Size, gen);

    float *d_in1;
    check_cuda( cudaMalloc(&d_in1, Size * sizeof(float)) );

    float *d_in2;
    check_cuda( cudaMalloc(&d_in2, Size * sizeof(float)) );

    float *d_out;
    check_cuda( cudaMalloc(&d_out, OutputSize * sizeof(float)) );

    check_cuda( cudaMemcpy(d_in1 , in1 , Size * sizeof(float), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy(d_in2 , in2 , Size * sizeof(float), cudaMemcpyHostToDevice ) );

    // CPU reference
    auto solution = convolution(in1, in2, Size);

    // GPU
    cuAlgo::fft1dCT_plan<OutputSize / 2>();
    cuAlgo::ifft1dCT_plan<OutputSize / 2>();
    cuAlgo::convFFT1dCT<OutputSize, BlockSize>(d_in1, d_in2, d_out, Size, Size, 1);

    check_cuda( cudaMemcpy ( out, d_out, OutputSize * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < OutputSize; ++i)
    {
        // std::cout << i << ": " << solution[i] << " --- " << out[i] << " --- "
        //     << std::abs(solution[i] - out[i]) / std::abs(solution[i])
        //     << std::endl;
        if(solution[i] > 1e-5 && out[i] > 1e-5)
            ASSERT_TRUE(std::abs(solution[i] - out[i]) / std::abs(solution[i]) < 1e-3);
    }

    check_cuda( cudaFree(d_in1) );
    check_cuda( cudaFree(d_in2) );
    check_cuda( cudaFree(d_out) );
    free(in1);
    free(in2);
    free(out);
}

TEST(CT_convFFT, size_256) {
    // Radix2
    constexpr unsigned int size = 256;
    run_single_test<size, 128>();
}

TEST(CT_convFFT, size_512) {
    // Radix2
    constexpr unsigned int size = 512;
    run_single_test<size, 128>();
}

TEST(CT_convFFT, size_1024) {
    // Radix2
    constexpr unsigned int size = 1024;
    run_single_test<size, 128>();
}

TEST(CT_convFFT, size_2048) {
    // Radix2
    constexpr unsigned int size = 2048;
    run_single_test<size, 128>();
}
