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
#include "cuAlgo/API/convFFT1dBailey.hpp"

using std::vector;
using std::swap;

// Bit reversal permutation
void bitReverse(vector<float2> &a)
{
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
void fft(vector<float2> &a, bool invert)
{
    int n = a.size();
    bitReverse(a);

    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * M_PI / len * (invert ? 1 : -1);

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

void ifft_1d(float2* x, int n)
{
    // --- Bit reversal ---
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if (i < j) {
            float2 tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }

    // --- FFT stages ---
    for (int len = 2; len <= n; len <<= 1) {
        float2 wlen = itwiddle(1, len);

        for (int i = 0; i < n; i += len) {
            float2 w = make_float2(1.0f, 0.0f);

            for (int j = 0; j < len/2; j++) {
                float2 u = x[i+j];
                float2 v = cmul(x[i+j+len/2], w);

                x[i+j] = cadd(u, v);
                x[i+j+len/2] = csub(u, v);

                w = cmul(w, wlen);
            }
        }
    }
}

void ifft_4step(float2* x, int N, int N1, int N2)
{
    std::vector<float2> tmp(std::max(N1, N2));

    // -------------------------------------------------
    // STEP 1: Row FFTs (size N2)
    // -------------------------------------------------
    for (int r = 0; r < N1; r++) {
        for (int c = 0; c < N2; c++)
            tmp[c] = x[r + c*N1];

        ifft_1d(tmp.data(), N2);

        // if(r == 0)
        //     std::cout << tmp[1].x << " " << tmp[1].y << std::endl;

        for (int c = 0; c < N2; c++)
            x[r + c*N1] = tmp[c];
    }

    // -------------------------------------------------
    // STEP 2: Twiddle multiply W_N^(r*c)
    // -------------------------------------------------
    for (int r = 0; r < N1; r++)
        for (int c = 0; c < N2; c++)
            x[r + c*N1] = cmul(x[r + c*N1], itwiddle(r * c, N));

    // -------------------------------------------------
    // STEP 3: Column FFTs (size N1)
    // -------------------------------------------------
    for (int c = 0; c < N2; c++) {
        for (int r = 0; r < N1; r++)
            tmp[r] = x[r + c*N1];

        ifft_1d(tmp.data(), N1);

        for (int r = 0; r < N1; r++)
            x[r + c*N1] = tmp[r];
    }

    // -------------------------------------------------
    // STEP 4: Final reorder (mixed-radix -> linear order)
    // -------------------------------------------------
    std::vector<float2> B(N);
    for (int r = 0; r < N1; r++)
        for (int c = 0; c < N2; c++)
            B[c + r*N2] = x[r + c*N1];

    for (int i = 0; i < N; i++)
        x[i] = make_float2(B[i].x / N, B[i].y / N);
}


// Convolution using FFT
vector<float2> convolution(const float2 * a, const float2 * b, int N) {
    int n = 1;
    while (n < N + N) n <<= 1;

    std::cout << "n = " << n << std::endl;

    vector<float2> fa(n), fb(n);
    for (size_t i = 0; i < N; i++) fa[i] = a[i];
    // for (size_t i = N; i < n; i++) fa[i] = make_float2(0.f, 0.f);
    for (size_t i = 0; i < N; i++) fb[i] = b[i];
    // for (size_t i = N; i < n; i++) fb[i] = make_float2(0.f, 0.f);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++) fa[i] = cmul(fa[i], fb[i]);
    fft(fa, true);
    // ifft_4step(fa.data(), n, 256, 4);

    vector<float2> result(n);
    for (size_t i = 0; i < result.size(); i++) result[i] = fa[i];
    return result;
}

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

template<unsigned int Size, unsigned int BlockSize>
void run_single_test()
{
    constexpr int OutputSize = next_pow2(2 * Size - 1);

    float2 * in1 = (float2*)malloc(Size * sizeof(float2));
    float2 * in2 = (float2*)malloc(Size * sizeof(float2));
    float2 * out = (float2*)malloc(OutputSize * sizeof(float2));

    // First create an instance of an engine.
    // std::random_device rnd_device;
    constexpr unsigned int seed = 12345;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{seed}; // Generates random integers
    std::uniform_int_distribution<> dist{1, 4};

    auto gen = [&]() { return dist(mersenne_engine); };

    std::generate((float *)in1, (float*)in1 + 2 * Size, gen);
    std::generate((float *)in2, (float*)in2 + 2 * Size, gen);

    float2 *d_in1;
    check_cuda( cudaMalloc(&d_in1, Size * sizeof(float2)) );

    float2 *d_in2;
    check_cuda( cudaMalloc(&d_in2, Size * sizeof(float2)) );

    float2 *d_out;
    check_cuda( cudaMalloc(&d_out, OutputSize * sizeof(float2)) );

    check_cuda( cudaMemcpy(d_in1 , in1 , Size * sizeof(float2), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy(d_in2 , in2 , Size * sizeof(float2), cudaMemcpyHostToDevice ) );

    // CPU reference
    auto solution = convolution(in1, in2, Size);

    // GPU
    // cuAlgo::conv1dCT_plan<OutputSize / 2>();
    cuAlgo::convFFT1dBailey<4, OutputSize / 4, Size, Size, BlockSize>(d_in1, d_in2, d_out, 1);

    check_cuda( cudaMemcpy ( out, d_out, OutputSize * sizeof(float2), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < OutputSize; ++i)
    {
        std::cout << i << ": " << solution[i].x << " --- " << out[i].x << " --- "
            << std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x)
            << std::endl;
        std::cout << i << ": " << solution[i].y << " --- " << out[i].y << " --- "
            << std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y)
            << std::endl;
            if(solution[i].x > 1e-5 && out[i].x > 1e-5)
                ASSERT_TRUE(std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x) < 2e-3);
            if(solution[i].y > 1e-5 && out[i].y > 1e-5)
                ASSERT_TRUE(std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y) < 2e-3);
    }

    check_cuda( cudaFree(d_in1) );
    check_cuda( cudaFree(d_in2) );
    check_cuda( cudaFree(d_out) );
    free(in1);
    free(in2);
    free(out);
}

// TEST(CT_convFFT, size_256) {
//     // Radix2
//     constexpr unsigned int size = 256;
//     run_single_test<size, 128>();
// }

TEST(CT_convFFT, size_512) {
    // Radix2
    constexpr unsigned int size = 512;
    run_single_test<size, 64>();
}

// TEST(CT_convFFT, size_1024) {
//     // Radix2
//     constexpr unsigned int size = 1024;
//     run_single_test<size, 128>();
// }

// TEST(CT_convFFT, size_2048) {
//     // Radix2
//     constexpr unsigned int size = 2048;
//     run_single_test<size, 128>();
// }

// TEST(CT_convFFT, size_4096) {
//     // Radix2
//     constexpr unsigned int size = 4096;
//     run_single_test<size, 256>();
// }

// TEST(CT_convFFT, size_8192) {
//     // Radix2
//     constexpr unsigned int size = 8192;
//     run_single_test<size, 256>();
// }
