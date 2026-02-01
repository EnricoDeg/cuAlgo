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

#define SQRT1_2 0.7071067811865476f

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

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 mul_neg_j(float2 a) {
    return make_float2(a.y, -a.x);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 mul_W8_1(float2 a) {
    return make_float2(
        SQRT1_2 * (a.x + a.y),
        SQRT1_2 * (a.y - a.x)
    );
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 mul_W8_3(float2 a) {
    return make_float2(
        -SQRT1_2 * (a.x - a.y),
        -SQRT1_2 * (a.x + a.y)
    );
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 conjf2(float2 a) {
    return make_float2(a.x, -a.y);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 cmulj(float2 a) {
    // multiply by j = i (complex unit)
    return make(-a.y, a.x);
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
float2 twiddle(int k, int m) {
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

CUALGO_DEVICE CUALGO_FORCE_INLINE
unsigned int base8_reverse(unsigned x, int log8N)
{
    unsigned r = 0;
    #pragma unroll
    for (int i = 0; i < log8N; ++i) {
        r = (r << 3) | (x & 0x7);
        x >>= 3;
    }
    return r;
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
unsigned int base_reverse(unsigned x, int log4N)
{
    unsigned r = 0;
    #pragma unroll
    for (int i = 0; i < log4N; ++i) {
        r = (r << 2) | (x & 0x3);
        x >>= 2;
    }
    return r;
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
unsigned int mixed_radix_reverse(unsigned x, int log4N)
{
    // total bits = 2*k + 1
    unsigned out = 0;
    unsigned bitpos = 2 * log4N;  // MSB position in output

    // 1) extract radix-2 digit (LSB)
    unsigned d0 = x & 0x1;
    out |= d0 << bitpos;
    bitpos -= 2;

    // 2) extract and reverse radix-4 digits
    x >>= 1;  // consume radix-2 digit

    for (int s = 0; s < log4N; ++s) {
        unsigned d = x & 0x3;   // radix-4 digit
        out |= d << bitpos;
        bitpos -= 2;
        x >>= 2;
    }

    return out;
}

template<typename HandleType>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void radix8_butterfly(
    float2* x,
    int stride,
    int tw,
    int N)
{
    float2 a0 = x[0];
    float2 a1 = cmul(x[stride],     HandleType::twiddles()[tw * 1]);
    float2 a2 = cmul(x[2 * stride], HandleType::twiddles()[tw * 2]);
    float2 a3 = cmul(x[3 * stride], HandleType::twiddles()[tw * 3]);
    float2 a4 = cmul(x[4 * stride], HandleType::twiddles()[tw * 4]);
    float2 a5 = cmul(x[5 * stride], HandleType::twiddles()[tw * 5]);
    float2 a6 = cmul(x[6 * stride], HandleType::twiddles()[tw * 6]);
    float2 a7 = cmul(x[7 * stride], HandleType::twiddles()[tw * 7]);

    float2 s0 = cadd(a0, a4);
    float2 s1 = cadd(a1, a5);
    float2 s2 = cadd(a2, a6);
    float2 s3 = cadd(a3, a7);

    float2 d0 = csub(a0, a4);
    float2 d1 = csub(a1, a5);
    float2 d2 = csub(a2, a6);
    float2 d3 = csub(a3, a7);

    float2 t0 = cadd(s0, s2);
    float2 t1 = cadd(s1, s3);
    float2 t2 = csub(s0, s2);
    float2 t3 = mul_neg_j(csub(s1, s3));

    float2 u0 = cadd(d0, mul_neg_j(d2));
    float2 u1 = cadd(d1, mul_neg_j(d3));
    float2 u2 = csub(d0, mul_neg_j(d2));
    float2 u3 = csub(d1, mul_neg_j(d3));

    x[0]           = cadd(t0, t1);
    x[stride]      = cadd(u0, mul_W8_1(u1));
    x[2 * stride]  = cadd(t2, t3);
    x[3 * stride]  = cadd(u2, mul_W8_3(u3));
    x[4 * stride]  = csub(t0, t1);
    x[5 * stride]  = csub(u0, mul_W8_1(u1));
    x[6 * stride]  = csub(t2, t3);
    x[7 * stride]  = csub(u2, mul_W8_3(u3));
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void radix2_CT(T* sdata, int tid, const int LOGN)
{
    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    for (int stage = 0, len = 2; stage < LOGN; ++stage, len <<= 1) {

        int half = len >> 1;

        for (int tidx = tid; tidx < FFTSize / 2; tidx += BlockSize)
        {
            int block = tidx / half; //>> (__ffs(len) - 2); // tidx / half;
            int k = tidx % half; //& (half - 1); //tidx % half;
            int i = block * len + k;

            int twiddle_idx = (k * FFTSize) / len;
            T w = HandleType::twiddles()[twiddle_idx];
            T u = sdata[i];
            T v = cmul(w, sdata[i + half]);

            sdata[i]       = cadd(u, v);
            sdata[i + half]= csub(u, v);
        }
        __syncthreads();
    }
}

#endif
