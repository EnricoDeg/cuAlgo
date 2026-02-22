/*
 * @file test_fft1dCT.cu
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
#include "cuAlgo/API/fft1dBailey.hpp"

void fft_1d(float2* x, int n)
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
        float2 wlen = twiddle(1, len);

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

void fft_4step(float2* x, int N, int N1, int N2)
{
    std::vector<float2> tmp(std::max(N1, N2));

    // -------------------------------------------------
    // STEP 1: Row FFTs (size N2)
    // -------------------------------------------------
    for (int r = 0; r < N1; r++) {
        for (int c = 0; c < N2; c++)
            tmp[c] = x[r + c*N1];

        fft_1d(tmp.data(), N2);

        for (int c = 0; c < N2; c++)
            x[r + c*N1] = tmp[c];
    }

    // -------------------------------------------------
    // STEP 2: Twiddle multiply W_N^(r*c)
    // -------------------------------------------------
    for (int r = 0; r < N1; r++)
        for (int c = 0; c < N2; c++)
            x[r + c*N1] = cmul(x[r + c*N1], twiddle(r * c, N));

    // -------------------------------------------------
    // STEP 3: Column FFTs (size N1)
    // -------------------------------------------------
    for (int c = 0; c < N2; c++) {
        for (int r = 0; r < N1; r++)
            tmp[r] = x[r + c*N1];

        fft_1d(tmp.data(), N1);

        for (int r = 0; r < N1; r++)
            x[r + c*N1] = tmp[r];
    }

    // // -------------------------------------------------
    // // STEP 4: Final reorder (mixed-radix -> linear order)
    // // -------------------------------------------------
    std::vector<float2> B(N);
    for (int r = 0; r < N1; r++)
        for (int c = 0; c < N2; c++)
            B[c + r*N2] = x[r + c*N1];

    for (int i = 0; i < N; i++)
        x[i] = B[i];
}

template<unsigned int Size1, unsigned int Size2, unsigned int BlockSize>
void run_single_test()
{
    constexpr unsigned int Size = Size1 * Size2;
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
    fft_4step(solution, Size, Size2, Size1);

    // GPU
    cuAlgo::fft1dCT_plan<Size2>();
    cuAlgo::fft1dBailey<Size1, Size2, BlockSize>(d_in, d_out, 1);

    check_cuda( cudaMemcpy ( out, d_out, Size * sizeof(float2), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < Size; ++i)
    {
        // std::cout << i << ": " << solution[i].x << " --- " << out[i].x << " --- "
        //     << std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x)
        //     << std::endl;
        // std::cout << i << ": " << solution[i].y << " --- " << out[i].y << " --- "
        //     << std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y)
        //     << std::endl;
        ASSERT_TRUE(std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x) < 2e-3);
        ASSERT_TRUE(std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y) < 2e-3);
    }

    check_cuda( cudaFree(d_in) );
    check_cuda( cudaFree(d_out) );
    free(in);
    free(out);
    free(solution);
}

TEST(CT_FFT, size_1024) {
    // Radix4
    constexpr unsigned int size = 1024;
    constexpr unsigned int size1 = 4;
    constexpr unsigned int size2 = size / size1;
    run_single_test<size1, size2, 64>();
}
