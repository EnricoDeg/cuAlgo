/*
 * @file test_batchNormFwdTraining.cu
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
#include <chrono>
#include <gtest/gtest.h>
#include "cuAlgo/API/batchNormFwdTraining.hpp"

using namespace std::chrono;

TEST(batchNormFwdTraining, default_values) {

    unsigned int N = 1;
    unsigned int C = 2;
    unsigned int HW = 1024;

    float * scale = (float *)malloc(C * sizeof(float));
    float * bias  = (float *)malloc(C * sizeof(float));
    for (unsigned int i = 0 ; i < C ; ++i) {
        scale[i] = 1.0;
        bias[i]  = 0.0;
    }

    float * in = (float *)malloc(N * C * HW * sizeof(float));
    for (unsigned int i = 0 ; i < N ; ++i)
        for (unsigned int j = 0 ; j < HW ; ++j)
            for (unsigned int k = 0 ; k < C ; ++k)
                in[k + j * C + i * C * HW] = (float)j / HW;

    float * out = (float *)malloc(N * C * HW * sizeof(float));
    float * solution = (float *)malloc(N * C * HW * sizeof(float));

    float *d_in;
    check_cuda( cudaMalloc(&d_in, N * C * HW * sizeof(float)) );

    float *d_out;
    check_cuda( cudaMalloc(&d_out, N * C * HW * sizeof(float)) );

    float *d_scale;
    check_cuda( cudaMalloc(&d_scale, C * sizeof(float)) );

    float *d_bias;
    check_cuda( cudaMalloc(&d_bias , C * sizeof(float)) );

    check_cuda( cudaMemcpy ( d_in   , in   , N * C * HW * sizeof(float), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy ( d_scale, scale, C * sizeof(float), cudaMemcpyHostToDevice ) );
    check_cuda( cudaMemcpy ( d_bias , bias , C * sizeof(float), cudaMemcpyHostToDevice ) );

    for (unsigned int k = 0 ; k < C ; ++k) {

        float mean = 0.0;
        float variance = 0.0;
        for (unsigned int i = 0 ; i < N ; ++i) {
            for (unsigned int j = 0 ; j < HW ; ++j) {
                float value = in[k + j * C + i * C * HW];
                mean += value;
                variance += (value * value);
            }
        }

        float epsilon = 1e-7;
        mean /= (N * HW);
        variance /= (N * HW);
        variance += (-mean * mean);
        float invVar = 1.0 / sqrt(variance + epsilon);

        for (unsigned int i = 0 ; i < N ; ++i) {
            for (unsigned int j = 0 ; j < HW ; ++j) {
                float elemStd = in[k + j * C + i * C * HW] - mean;
                solution[k + j * C + i * C * HW] = scale[k] * (invVar * elemStd) + bias[k];
            }
        }
    }

    cuAlgo::batchNormFwdTraining<1024U, 32U, true>(d_in, d_out, d_scale, d_bias, N, C, HW);

    check_cuda( cudaMemcpy ( out, d_out, N * C * HW * sizeof(float), cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0 ; i < N ; ++i)
        for (unsigned int j = 0 ; j < HW ; ++j)
            for (unsigned int k = 0 ; k < C ; ++k)
                ASSERT_TRUE((solution[k + j * C + i * C * HW] - out[k + j * C + i * C * HW]) /
                    solution[k + j * C + i * C * HW] < 1e-5);

    check_cuda( cudaFree(d_in) );
    check_cuda( cudaFree(d_out) );
    check_cuda( cudaFree(d_scale) );
    check_cuda( cudaFree(d_bias) );
    free(in);
    free(out);
    free(scale);
    free(bias);
    free(solution);
}
