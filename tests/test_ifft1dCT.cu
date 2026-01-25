/*
 * @file test_ifft1dCT.cu
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
#include "cuAlgo/API/ifft1dCT.hpp"

void bit_reverse_permute(float2 * a, unsigned int n) {
    size_t j = 0;

    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;

        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
}

void ifft_cpu(float2* a, unsigned int n) {

    assert((n & (n - 1)) == 0);

    // 1. Bit-reversal reorder
    bit_reverse_permute(a, n);

    // 2. FFT stages
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half_len = len >> 1;

        // Inverse FFT uses + angle
        float angle = 2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        float2 w_len = { std::cos(angle), std::sin(angle) }; // e^{+jθ}

        for (size_t i = 0; i < n; i += len) {
            float2 w = {1.0f, 0.0f};

            for (size_t j = 0; j < half_len; ++j) {
                float2 u = a[i + j];
                float2 t = cmul(w, a[i + j + half_len]);

                a[i + j]             = cadd(u, t);
                a[i + j + half_len]  = csub(u, t);

                w = cmul(w, w_len);
            }
        }
    }

    // 3. Scale by 1/N
    float inv_n = 1.0f / static_cast<float>(n);
    for(int i = 0; i < n; ++i)
    {
        a[i].x *= inv_n;
        a[i].y *= inv_n;
    }
}

template<unsigned int Size, unsigned int BlockSize>
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
    ifft_cpu(solution, Size);

    // GPU
    cuAlgo::ifft1dCT_plan<Size>();
    cuAlgo::ifft1dCT<Size, BlockSize>(d_in, d_out, 1);

    check_cuda( cudaMemcpy ( out, d_out, Size * sizeof(float2), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < Size; ++i)
    {
        // std::cout << i << ": " << solution[i].x << " --- " << out[i].x << " --- "
        //     << std::abs(solution[i].x - out[i].x) / std::abs(solution[i].x)
        //     << std::endl;
        // std::cout << i << ": " << solution[i].y << " --- " << out[i].y << " --- "
        //     << std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y)
        //     << std::endl;
        if(solution[i].x > 1e-5 && out[i].x > 1e-5)
            ASSERT_TRUE(std::abs(solution[i].x -out[i].x) / std::abs(solution[i].x) < 1e-3);
        if(solution[i].y > 1e-5 && out[i].y > 1e-5)
            ASSERT_TRUE(std::abs(solution[i].y - out[i].y) / std::abs(solution[i].y) < 1e-3);
    }

    check_cuda( cudaFree(d_in) );
    check_cuda( cudaFree(d_out) );
    free(in);
    free(out);
    free(solution);
}

TEST(CT_IFFT, size_1024) {
    // Radix2
    constexpr unsigned int size = 1024;
    run_single_test<size, 128>();
}
