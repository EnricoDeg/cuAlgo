/*
 * @file fft.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
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

#ifndef FFT_HPP
#define FFT_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 make(float a, float b) {
    return make_float2(a, b);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 cmul(float2 a, float2 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__ float2 twiddle(int k, int m) {
    float angle = -2.0f * M_PI * k / m;
    return make_float2(cosf(angle), sinf(angle));
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
unsigned int base2_reverse(unsigned x, int log2N)
{
    unsigned r = 0;
    #pragma unroll
    for (int i = 0; i < log2N; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
unsigned int base4_reverse(unsigned x, int log4N)
{
    unsigned r = 0;
    #pragma unroll
    for (int i = 0; i < log4N; ++i) {
        r = (r << 2) | (x & 0x3);
        x >>= 2;
    }
    return r;
}

#endif
