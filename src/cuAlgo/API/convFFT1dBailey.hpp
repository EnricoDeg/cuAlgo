/*
 * @file convFFT1dBailey.hpp
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

#ifndef CONVFFT1DBAILEY_HPP
#define CONVFFT1DBAILEY_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"
#include "cuAlgo/API/fft1dPlan.hpp"

#define BANKS 32

// -------------------------------------------------------------------------------------------------
// Opimized implementation
// -------------------------------------------------------------------------------------------------
CUALGO_DEVICE CUALGO_FORCE_INLINE
int permute(int idx)
{
    // int row = idx / 32;
    // int col = idx & (32 - 1);
    // return row * 32 + (col ^ row);
    return idx + (idx >> 5);   // i / 32
}

#define PAD_T 1

template<unsigned int FFTSize1>
CUALGO_DEVICE CUALGO_FORCE_INLINE
int pad_transpose(int row, int col) {
    // row = thread index
    // col = FFTSize1 index
    return row * (FFTSize1 + PAD_T) + col;
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
float GET_INV(int i) {
    constexpr float table[10] = {
        1.570796327f,   // 2*pi / 4
        0.392699082f,   // 2*pi / 16
        0.098174770f,   // 2*pi / 64
        0.024543693f,   // 2*pi / 256
        0.006135923f,   // 2*pi / 1024
        0.001533981f,   // 2*pi / 4096
        0.000383495f,   // 2*pi / 16384
        0.000095873f,   // 2*pi / 65536
        0.000023968f,   // 2*pi / 262144
        0.000005992f    // 2*pi / 1048576
    };
    return table[i];
}

template<unsigned int FFTSize, typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fftN_reg_fast(Columns*... cols);

// specialized for radix-4 (1 step in register)
template<unsigned int FFTSize>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fftN_reg_fast(float2 *a0, float2 *a1, float2 *a2, float2 *a3,
                   float2 *b0, float2 *b1, float2 *b2, float2 *b3)
{
    // Radix4 in register

    // First signal
    float2 x0 = *a0;
    float2 x1 = *a1;
    float2 x2 = *a2;
    float2 x3 = *a3;

    float2 t0 = cadd(x0, x2);
    float2 t1 = cadd(x1, x3);
    *a0 = cadd(t0, t1);
    *a2 = csub(t0, t1);

    t0 = csub(x0, x2);
    t1 = csub(x1, x3);

    *a1 = make(t0.x + t1.y, t0.y - t1.x);
    *a3 = make(t0.x - t1.y, t0.y + t1.x);

    // Second signal
    x0 = *b0;
    x1 = *b1;
    x2 = *b2;
    x3 = *b3;

    t0 = cadd(x0, x2);
    t1 = cadd(x1, x3);

    *b0 = cadd(t0, t1);
    *b2 = csub(t0, t1);

    t0 = csub(x0, x2);
    t1 = csub(x1, x3);

    *b1 = make(t0.x + t1.y, t0.y - t1.x);
    *b3 = make(t0.x - t1.y, t0.y + t1.x);
}

template<unsigned int FFTSize, typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void ifftN_reg_fast(Columns*... cols);

// specialized for radix-4 (1 step in register)
template<unsigned int FFTSize>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void ifftN_reg_fast(float2 *a0, float2 *a1, float2 *a2, float2 *a3)
{
    // Radix4 in register
    float2 x0 = *a0;
    float2 x1 = *a1;
    float2 x2 = *a2;
    float2 x3 = *a3;

    float2 t0 = cadd(x0, x2);
    float2 t1 = cadd(x1, x3);

    *a0 = cadd(t0, t1);
    *a2 = csub(t0, t1);

    t0 = csub(x0, x2);
    t1 = csub(x1, x3);

    *a1 = make(t0.x - t1.y, t0.y + t1.x);
    *a3 = make(t0.x + t1.y, t0.y - t1.x);
}

template<typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void twiddle_stagex2(float2 alpha, Columns* CUALGO_RESTRICT ... cols)
{
    constexpr int N = sizeof...(Columns);
    float2* ptrs[N] = { cols... };  // turn parameter pack into array for static-for

    // initial twiddle w = 1 + 0i
    float2 w = make_float2(1.f, 0.f);

    {
        // update w = w * alpha using fused multiplies
        float wx = __fmaf_rn(-w.y, alpha.y, w.x * alpha.x); // w.x*alpha.x - w.y*alpha.y
        float wy = __fmaf_rn(w.x, alpha.y, w.y * alpha.x);  // w.x*alpha.y + w.y*alpha.x
        w.x = wx;
        w.y = wy;
    }

    // unrolled loop over all columns
    static_for<1, N / 2>([&](auto I){
        // load current element
        float2 ai = *ptrs[I.value];
        float2 bi = *ptrs[I.value + N / 2];

        // fused complex multiply: ai * w
        float real1 = __fmaf_rn(-ai.y, w.y, ai.x * w.x); // ai.x*w.x - ai.y*w.y
        float imag1 = __fmaf_rn( ai.x, w.y, ai.y * w.x);  // ai.x*w.y + ai.y*w.x

        float real2 = __fmaf_rn(-bi.y, w.y, bi.x * w.x); // ai.x*w.x - ai.y*w.y
        float imag2 = __fmaf_rn( bi.x, w.y, bi.y * w.x);  // ai.x*w.y + ai.y*w.x

        // store result
        *ptrs[I.value]         = make_float2(real1, imag1);
        *ptrs[I.value + N / 2] = make_float2(real2, imag2);

        // update w = w * alpha for next element
        float wx = __fmaf_rn(-w.y, alpha.y, w.x * alpha.x);
        float wy = __fmaf_rn( w.x, alpha.y, w.y * alpha.x);
        w.x = wx;
        w.y = wy;
    });
}

template<typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void twiddle_stage(float2 alpha, Columns* CUALGO_RESTRICT ... cols)
{
    constexpr int N = sizeof...(Columns);
    float2* ptrs[N] = { cols... };  // turn parameter pack into array for static-for

    // initial twiddle w = 1 + 0i
    float2 w = make_float2(1.f, 0.f);

    {
        // update w = w * alpha using fused multiplies
        float wx = __fmaf_rn(-w.y, alpha.y, w.x * alpha.x); // w.x*alpha.x - w.y*alpha.y
        float wy = __fmaf_rn(w.x, alpha.y, w.y * alpha.x);  // w.x*alpha.y + w.y*alpha.x
        w.x = wx;
        w.y = wy;
    }

    // unrolled loop over all columns
    static_for<1, N>([&](auto I){
        // load current element
        float2 ai = *ptrs[I.value];

        // fused complex multiply: ai * w
        float real = __fmaf_rn(-ai.y, w.y, ai.x * w.x); // ai.x*w.x - ai.y*w.y
        float imag = __fmaf_rn(ai.x, w.y, ai.y * w.x);  // ai.x*w.y + ai.y*w.x

        // store result
        *ptrs[I.value] = make_float2(real, imag);

        // update w = w * alpha for next element
        float wx = __fmaf_rn(-w.y, alpha.y, w.x * alpha.x);
        float wy = __fmaf_rn(w.x, alpha.y, w.y * alpha.x);
        w.x = wx;
        w.y = wy;
    });
}

template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    unsigned int IDataSize,
    typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage1(float2* CUALGO_RESTRICT idata1,
            float2* CUALGO_RESTRICT idata2,
            Columns* CUALGO_RESTRICT... columns)
{
    int tid = threadIdx.x;
    constexpr int N = sizeof...(Columns);

    // Turn parameter pack into an array of pointers
    float2* ptrs[N] = { columns... };

    // Compile-time unrolled assignment using static_for
    static_for<0, N / 2>([&](auto I){
        if constexpr(It * BlockSize + FFTSize2 * I.value + BlockSize > IDataSize)
        {
            *ptrs[I.value] = make(0.f, 0.f);
        }
        else
        {
            *ptrs[I.value] = *(idata1 + tid + It * BlockSize + FFTSize2 * I.value);
        }
    });

    static_for<0, N / 2>([&](auto I){
        if constexpr(It * BlockSize + FFTSize2 * I.value + BlockSize > IDataSize)
        {
            *ptrs[I.value + N / 2] = make(0.f, 0.f);
        }
        else
        {
            *ptrs[I.value + N / 2] = *(idata2 + tid + It * BlockSize + FFTSize2 * I.value);
        }
    });

    // --------------------------
    // STEP 1: FFT on columns
    // --------------------------
    fftN_reg_fast<FFTSize1>(columns...);

    float inv = 1.f / (FFTSize2 * FFTSize1);
    // --------------------------
    // STEP 2: Multiply twiddle factors
    // W16^(n1 * n2)
    // Store back in strided layout
    // --------------------------
    {
        float ang = -2.f * float(M_PI) *
            float(tid + It * BlockSize) * inv;
        float s, c;
        __sincosf(ang, &s, &c);
        float2 alpha = make_float2(c, s);
        twiddle_stagex2(alpha, columns...);
    }
}

template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage3(Columns* ... columns)
{
    int tid = threadIdx.x;

    // --------------------------
    // STEP 1: FFT on columns
    // --------------------------
    ifftN_reg_fast<FFTSize1>(columns...);

    float inv = 1.f / (FFTSize2 * FFTSize1);
    // --------------------------
    // STEP 2: Multiply twiddle factors
    // W16^(n1 * n2)
    // Store back in strided layout
    // --------------------------
    {
        float ang = 2.f * float(M_PI) *
            float(tid + It * BlockSize) * inv;
        float s, c;
        __sincosf(ang, &s, &c);
        float2 alpha = make_float2(c, s);
        twiddle_stage(alpha, columns...);
    }
}

template<
unsigned int It,
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
unsigned int Offset,
typename... Rows
>
__device__ void stage2(float* CUALGO_RESTRICT buf0,
                       float* CUALGO_RESTRICT buf1,
                       float* CUALGO_RESTRICT buf2,
                       float* CUALGO_RESTRICT buf3,
                       Rows* CUALGO_RESTRICT... rows)
{
    constexpr int LOG2N = __builtin_ctz(FFTSize2);
    constexpr int log4N = LOG2N >> 1;

    // Turn parameter pack into an array of pointers
    constexpr int N = sizeof...(Rows);
    float2* ptrs[N] = { rows... };

    int tid = threadIdx.x;
    int i = tid;

    float* in1  = buf0;
    float* out1 = buf1;

    float* in2  = buf2;
    float* out2 = buf3;

    constexpr int t = FFTSize2 >> 2;

    // first stage, all data that we need are in register already
    {
        constexpr int mh = 1;
        int k = i & (mh - 1);

        float inv =  GET_INV(0); // 1.f / (mh << 2);
        float ang = - float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        {
            // load inputs
            float2 x0, x1, x2, x3;
            x0.x = (*ptrs[0]).x;
            x1.x = (*ptrs[1]).x;
            x2.x = (*ptrs[2]).x;
            x3.x = (*ptrs[3]).x;

            x0.y = (*ptrs[0]).y;
            x1.y = (*ptrs[1]).y;
            x2.y = (*ptrs[2]).y;
            x3.y = (*ptrs[3]).y;

            // apply twiddle factors using __fmaf_rn
            float a0_real = x0.x;
            float a0_imag = x0.y;

            float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
            float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

            float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
            float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

            float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
            float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

            // butterflies
            float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
            float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

            float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
            float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

            // output indices
            int o = (i - k) * 4 + k;

            // first stage
            out1[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
            out1[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
            out1[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
            out1[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

            // second stage
            b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
            b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

            // mul_neg_j(csub(a1,a3)) -> fused
            b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
            b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

            out1[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
            out1[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
            out1[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
            out1[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        }

        {

            // load inputs
            float2 x0, x1, x2, x3;
            x0.x = (*ptrs[4]).x;
            x1.x = (*ptrs[5]).x;
            x2.x = (*ptrs[6]).x;
            x3.x = (*ptrs[7]).x;

            x0.y = (*ptrs[4]).y;
            x1.y = (*ptrs[5]).y;
            x2.y = (*ptrs[6]).y;
            x3.y = (*ptrs[7]).y;

            // apply twiddle factors using __fmaf_rn
            float a0_real = x0.x;
            float a0_imag = x0.y;

            float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
            float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

            float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
            float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

            float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
            float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

            // butterflies
            float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
            float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

            float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
            float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

            // output indices
            int o = (i - k) * 4 + k;

            // first stage
            out2[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
            out2[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
            out2[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
            out2[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

            // second stage
            b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
            b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

            // mul_neg_j(csub(a1,a3)) -> fused
            b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
            b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

            out2[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
            out2[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
            out2[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
            out2[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        }

        float* tmp1 = in1; in1 = out1; out1 = tmp1;
        float* tmp2 = in2; in2 = out2; out2 = tmp2;
    }

    __syncthreads();

    // steps 1..log4N-1
    static_for<1, log4N - 1>([&](auto s)
    {
        constexpr int mh = 1 << (2 * s.value);
        int k = i & (mh - 1);

        float inv =  GET_INV(s.value); // 1.f / (mh << 2);
        float ang = - float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        {
            // load inputs
            float2 x0, x1, x2, x3;
            x0.x = in1[permute(i + 0*t)];
            x1.x = in1[permute(i + 1*t)];
            x2.x = in1[permute(i + 2*t)];
            x3.x = in1[permute(i + 3*t)];

            x0.y = in1[permute(i + 0*t) + Offset];
            x1.y = in1[permute(i + 1*t) + Offset];
            x2.y = in1[permute(i + 2*t) + Offset];
            x3.y = in1[permute(i + 3*t) + Offset];

            // apply twiddle factors using __fmaf_rn
            float a0_real = x0.x;
            float a0_imag = x0.y;

            float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
            float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

            float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
            float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

            float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
            float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

            // butterflies
            float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
            float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

            float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
            float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

            // output indices
            int o = (i - k) * 4 + k;

            // first stage
            out1[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
            out1[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
            out1[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
            out1[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

            // second stage
            b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
            b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

            // mul_neg_j(csub(a1,a3)) -> fused
            b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
            b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

            out1[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
            out1[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
            out1[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
            out1[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        }

        {
            // load inputs
            float2 x0, x1, x2, x3;
            x0.x = in2[permute(i + 0*t)];
            x1.x = in2[permute(i + 1*t)];
            x2.x = in2[permute(i + 2*t)];
            x3.x = in2[permute(i + 3*t)];

            x0.y = in2[permute(i + 0*t) + Offset];
            x1.y = in2[permute(i + 1*t) + Offset];
            x2.y = in2[permute(i + 2*t) + Offset];
            x3.y = in2[permute(i + 3*t) + Offset];

            // apply twiddle factors using __fmaf_rn
            float a0_real = x0.x;
            float a0_imag = x0.y;

            float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
            float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

            float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
            float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

            float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
            float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

            // butterflies
            float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
            float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

            float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
            float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

            // output indices
            int o = (i - k) * 4 + k;

            // first stage
            out2[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
            out2[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
            out2[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
            out2[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

            // second stage
            b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
            b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

            // mul_neg_j(csub(a1,a3)) -> fused
            b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
            b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

            out2[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
            out2[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
            out2[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
            out2[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        }

        __syncthreads();
        float* tmp1 = in1; in1 = out1; out1 = tmp1;
        float* tmp2 = in2; in2 = out2; out2 = tmp2;
    });

    // last step, we don't store to shared memory but keep results in register
    {
        constexpr int mh = 1 << (2 * (log4N - 1));
        int k = i & (mh - 1);

        float inv =  GET_INV(log4N - 1); // 1.f / (mh << 2);
        float ang = - float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        // load inputs
        float2 x0, x1, x2, x3;
        x0.x = in1[permute(i + 0*t)];
        x1.x = in1[permute(i + 1*t)];
        x2.x = in1[permute(i + 2*t)];
        x3.x = in1[permute(i + 3*t)];

        x0.y = in1[permute(i + 0*t) + Offset];
        x1.y = in1[permute(i + 1*t) + Offset];
        x2.y = in1[permute(i + 2*t) + Offset];
        x3.y = in1[permute(i + 3*t) + Offset];

        // apply twiddle factors using __fmaf_rn
        float a0_real = x0.x;
        float a0_imag = x0.y;

        float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
        float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

        float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
        float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

        float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
        float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

        // butterflies
        float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
        float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

        float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
        float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

        // first stage
        (*ptrs[0]).x = __fmaf_rn( 1.f, b0_real, b2_real);
        (*ptrs[0]).y = __fmaf_rn( 1.f, b0_imag, b2_imag);
        (*ptrs[2]).x = __fmaf_rn(-1.f, b2_real, b0_real);
        (*ptrs[2]).y = __fmaf_rn(-1.f, b2_imag, b0_imag);

        // second stage
        b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
        b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

        // mul_neg_j(csub(a1,a3)) -> fused
        b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
        b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

        (*ptrs[1]).x = __fmaf_rn(1.f, b0_real,  b2_real);
        (*ptrs[1]).y = __fmaf_rn(1.f, b0_imag,  b2_imag);
        (*ptrs[3]).x = __fmaf_rn(1.f, b0_real, -b2_real);
        (*ptrs[3]).y = __fmaf_rn(1.f, b0_imag, -b2_imag);
    }


    {
        constexpr int mh = 1 << (2 * (log4N - 1));
        int k = i & (mh - 1);

        float inv =  GET_INV(log4N - 1); // 1.f / (mh << 2);
        float ang = - float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        // load inputs
        float2 x0, x1, x2, x3;
        x0.x = in2[permute(i + 0*t)];
        x1.x = in2[permute(i + 1*t)];
        x2.x = in2[permute(i + 2*t)];
        x3.x = in2[permute(i + 3*t)];

        x0.y = in2[permute(i + 0*t) + Offset];
        x1.y = in2[permute(i + 1*t) + Offset];
        x2.y = in2[permute(i + 2*t) + Offset];
        x3.y = in2[permute(i + 3*t) + Offset];

        // apply twiddle factors using __fmaf_rn
        float a0_real = x0.x;
        float a0_imag = x0.y;

        float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
        float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

        float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
        float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

        float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
        float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

        // butterflies
        float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
        float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

        float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
        float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

        // first stage
        (*ptrs[4]).x = __fmaf_rn( 1.f, b0_real, b2_real);
        (*ptrs[4]).y = __fmaf_rn( 1.f, b0_imag, b2_imag);
        (*ptrs[6]).x = __fmaf_rn(-1.f, b2_real, b0_real);
        (*ptrs[6]).y = __fmaf_rn(-1.f, b2_imag, b0_imag);

        // second stage
        b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
        b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

        // mul_neg_j(csub(a1,a3)) -> fused
        b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
        b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

        (*ptrs[5]).x = __fmaf_rn(1.f, b0_real,  b2_real);
        (*ptrs[5]).y = __fmaf_rn(1.f, b0_imag,  b2_imag);
        (*ptrs[7]).x = __fmaf_rn(1.f, b0_real, -b2_real);
        (*ptrs[7]).y = __fmaf_rn(1.f, b0_imag, -b2_imag);
    }
}

template<
unsigned int It,
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
unsigned int Offset,
typename... Rows
>
__device__ void stage4(float* CUALGO_RESTRICT buf0,
                       float* CUALGO_RESTRICT buf1,
                       Rows* CUALGO_RESTRICT... rows)
{
    constexpr int LOG2N = __builtin_ctz(FFTSize2);
    constexpr int log4N = LOG2N >> 1;

    // Turn parameter pack into an array of pointers
    constexpr int N = sizeof...(Rows);
    float2* ptrs[N] = { rows... };

    int tid = threadIdx.x;
    int i = tid;

    float* in  = buf0;
    float* out = buf1;

    constexpr int t = FFTSize2 >> 2;

    // first stage, all data that we need are in register already
    {
        constexpr int mh = 1;
        int k = i & (mh - 1);

        float inv = GET_INV(0); // 1.f / (mh << 2);
        float ang = float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        // load inputs
        float2 x0, x1, x2, x3;
        x0.x = (*ptrs[0]).x;
        x1.x = (*ptrs[1]).x;
        x2.x = (*ptrs[2]).x;
        x3.x = (*ptrs[3]).x;

        x0.y = (*ptrs[0]).y;
        x1.y = (*ptrs[1]).y;
        x2.y = (*ptrs[2]).y;
        x3.y = (*ptrs[3]).y;

        // apply twiddle factors using __fmaf_rn
        float a0_real = x0.x;
        float a0_imag = x0.y;

        float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
        float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

        float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
        float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

        float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
        float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

        // butterflies
        float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
        float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

        float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
        float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

        // output indices
        int o = (i - k) * 4 + k;

        // first stage
        out[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
        out[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
        out[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
        out[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

        // second stage
        b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
        b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

        // mul_neg_j(csub(a1,a3)) -> fused
        b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
        b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

        out[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
        out[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        out[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
        out[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);

        __syncthreads();
        float* tmp = in; in = out; out = tmp;
    }

    // steps 1..log4N-1
    static_for<1, log4N - 1>([&](auto s)
    {
        constexpr int mh = 1 << (2 * s.value);
        int k = i & (mh - 1);

        float inv = GET_INV(s.value); // 1.f / (mh << 2);
        float ang = float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        // load inputs
        float2 x0, x1, x2, x3;
        x0.x = in[permute(i + 0*t)];
        x1.x = in[permute(i + 1*t)];
        x2.x = in[permute(i + 2*t)];
        x3.x = in[permute(i + 3*t)];

        x0.y = in[permute(i + 0*t) + Offset];
        x1.y = in[permute(i + 1*t) + Offset];
        x2.y = in[permute(i + 2*t) + Offset];
        x3.y = in[permute(i + 3*t) + Offset];

        // apply twiddle factors using __fmaf_rn
        float a0_real = x0.x;
        float a0_imag = x0.y;

        float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
        float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

        float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
        float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

        float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
        float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

        // butterflies
        float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
        float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

        float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
        float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

        // output indices
        int o = (i - k) * 4 + k;

        // first stage
        out[permute(o + 0*mh)] = __fmaf_rn( 1.f, b0_real, b2_real);
        out[permute(o + 0*mh) + Offset] = __fmaf_rn( 1.f, b0_imag, b2_imag);
        out[permute(o + 2*mh)] = __fmaf_rn(-1.f, b2_real, b0_real);
        out[permute(o + 2*mh) + Offset] = __fmaf_rn(-1.f, b2_imag, b0_imag);

        // second stage
        b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
        b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

        // mul_neg_j(csub(a1,a3)) -> fused
        b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
        b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

        out[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
        out[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);
        out[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
        out[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);

        __syncthreads();
        float* tmp = in; in = out; out = tmp;
    });

    // last step, we don't store to shared memory but keep results in register
    {
        constexpr int mh = 1 << (2 * (log4N - 1));
        int k = i & (mh - 1);

        float inv = GET_INV(log4N - 1); // 1.f / (mh << 2);
        float ang = float(k) * inv;
        float t_imag, t_real;
        __sincosf(ang, &t_imag, &t_real);

        // twiddle factors
        float tw1_real = __fmaf_rn(t_real, t_real, -t_imag * t_imag);
        float tw1_imag = __fmaf_rn(2.f * t_real, t_imag, 0.f);

        float tw2_real = __fmaf_rn(tw1_real, t_real, -tw1_imag * t_imag);
        float tw2_imag = __fmaf_rn(tw1_real, t_imag, tw1_imag * t_real);

        // load inputs
        float2 x0, x1, x2, x3;
        x0.x = in[permute(i + 0*t)];
        x1.x = in[permute(i + 1*t)];
        x2.x = in[permute(i + 2*t)];
        x3.x = in[permute(i + 3*t)];

        x0.y = in[permute(i + 0*t) + Offset];
        x1.y = in[permute(i + 1*t) + Offset];
        x2.y = in[permute(i + 2*t) + Offset];
        x3.y = in[permute(i + 3*t) + Offset];

        // apply twiddle factors using __fmaf_rn
        float a0_real = x0.x;
        float a0_imag = x0.y;

        float a1_real = __fmaf_rn(t_real, x1.x, -t_imag * x1.y);
        float a1_imag = __fmaf_rn(t_real, x1.y, t_imag * x1.x);

        float a2_real = __fmaf_rn(tw1_real, x2.x, -tw1_imag * x2.y);
        float a2_imag = __fmaf_rn(tw1_real, x2.y, tw1_imag * x2.x);

        float a3_real = __fmaf_rn(tw2_real, x3.x, -tw2_imag * x3.y);
        float a3_imag = __fmaf_rn(tw2_real, x3.y, tw2_imag * x3.x);

        // butterflies
        float b0_real = __fmaf_rn(1.f, a0_real, a2_real); // a0_real + a2_real
        float b0_imag = __fmaf_rn(1.f, a0_imag, a2_imag); // a0_imag + a2_imag

        float b2_real = __fmaf_rn(1.f, a1_real, a3_real); // a1_real + a3_real
        float b2_imag = __fmaf_rn(1.f, a1_imag, a3_imag); // a1_imag + a3_imag

        // first stage
        (*ptrs[0]).x = __fmaf_rn( 1.f, b0_real, b2_real);
        (*ptrs[0]).y = __fmaf_rn( 1.f, b0_imag, b2_imag);
        (*ptrs[2]).x = __fmaf_rn(-1.f, b2_real, b0_real);
        (*ptrs[2]).y = __fmaf_rn(-1.f, b2_imag, b0_imag);

        // second stage
        b0_real = __fmaf_rn(1.f, a0_real, -a2_real); // a0_real - a2_real
        b0_imag = __fmaf_rn(1.f, a0_imag, -a2_imag); // a0_imag - a2_imag

        // mul_neg_j(csub(a1,a3)) -> fused
        b2_real = __fmaf_rn(1.f, a1_imag, -a3_imag); // a1_imag - a3_imag
        b2_imag = __fmaf_rn(1.f, a3_real, -a1_real); // a3_real - a1_real

        (*ptrs[1]).x = __fmaf_rn(1.f, b0_real, -b2_real);
        (*ptrs[1]).y = __fmaf_rn(1.f, b0_imag, -b2_imag);
        (*ptrs[3]).x = __fmaf_rn(1.f, b0_real,  b2_real);
        (*ptrs[3]).y = __fmaf_rn(1.f, b0_imag,  b2_imag);
    }
}

// Get pointers to columns at row Idx
template<int NumCols, int NumRows, int Idx>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void get_column_ptrs(float2* out[NumCols], float2 (&a)[NumCols][NumRows])
{
    static_for<0, NumCols>([&](auto j){
        out[j.value] = &a[j.value][Idx];
    });
}

// Get pointers to rows at column Idx
template<int NumCols, int NumRows, int Idx>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void get_row_ptrs(float2* out[NumRows], float2 (&a)[NumCols][NumRows])
{
    static_for<0, NumRows>([&](auto j){
        out[j.value] = &a[Idx][j.value];
    });
}

// Stage1 wrapper for any number of columns
template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    unsigned int IDataSize,
    int NumCols,
    int NumRows
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage1_static_columns(float2* idata1,
                           float2* idata2,
                           float2 (&a)[NumCols][NumRows],
                           float2 (&b)[NumCols][NumRows])
{
    float2* cols[2 * NumCols];
    float2* cols_a[NumCols];
    get_column_ptrs<NumCols, NumRows, It>(cols_a, a);
    float2* cols_b[NumCols];
    get_column_ptrs<NumCols, NumRows, It>(cols_b, b);
    static_for<0, NumCols>([&](auto I){
        cols[I.value] = cols_a[I.value];
    });
    static_for<0, NumCols>([&](auto I){
        cols[I.value + NumCols] = cols_b[I.value];
    });

    expand_array(cols, [&](auto... args){
        stage1<It, FFTSize1, FFTSize2, BlockSize, IDataSize>(idata1, idata2, args...);
    }, std::make_integer_sequence<int, 2 * NumCols>{});
}

// Stage3 wrapper for any number of columns
template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    int NumCols,
    int NumRows
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage3_static_columns(float2 (&a)[NumCols][NumRows])
{
    float2* cols[NumCols];
    get_column_ptrs<NumCols, NumRows, It>(cols, a);

    expand_array(cols, [&](auto... args){
        stage3<It, FFTSize1, FFTSize2, BlockSize>(args...);
    }, std::make_integer_sequence<int, NumCols>{});
}

// Stage2 wrapper for any number of rows
template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    unsigned int Offset,
    int NumCols,
    int NumRows
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage2_static_row(float* buf0,
                       float* buf1,
                       float* buf2,
                       float* buf3,
                       float2 (&a)[NumCols][NumRows],
                       float2 (&b)[NumCols][NumRows])
{
    float2* rows[2 * NumRows];
    float2* rows_a[NumRows];
    get_row_ptrs<NumCols, NumRows, It>(rows_a, a);
    float2* rows_b[NumRows];
    get_row_ptrs<NumCols, NumRows, It>(rows_b, b);
    static_for<0, NumRows>([&](auto I){
        rows[I.value] = rows_a[I.value];
    });
    static_for<0, NumRows>([&](auto I){
        rows[I.value + NumRows] = rows_b[I.value];
    });

    // Expand array into variadic stage2
    expand_array(rows, [&](auto... args){
        stage2<It, FFTSize1, FFTSize2, BlockSize, Offset>(buf0, buf1, buf2, buf3, args...);
    }, std::make_integer_sequence<int, 2 * NumRows>{});
}

// Stage4 wrapper for any number of rows
template<
    unsigned int It,
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int BlockSize,
    unsigned int Offset,
    int NumCols,
    int NumRows
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage4_static_row(float* buf0,
                       float* buf1,
                       float2 (&a)[NumCols][NumRows])
{
    float2* rows[NumRows];
    get_row_ptrs<NumCols, NumRows, It>(rows, a);

    // Expand array into variadic stage4
    expand_array(rows, [&](auto... args){
        stage4<It, FFTSize1, FFTSize2, BlockSize, Offset>(buf0, buf1, args...);
    }, std::make_integer_sequence<int, NumRows>{});
}

template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int IDataSize1,
unsigned int IDataSize2,
unsigned int BlockSize,
typename T>
__global__ __launch_bounds__(BlockSize)
void convFFT1dBaileyKernel(T* CUALGO_RESTRICT input_data1,
                           T* CUALGO_RESTRICT input_data2,
                           T* CUALGO_RESTRICT output_data,
                           int batch_size)
{
    // add spec function for in register first stage fft
    static_assert(FFTSize1 == 4);
    static_assert(FFTSize2 <= 1048576);

    constexpr unsigned int FFTSize = FFTSize1 * FFTSize2;
    constexpr unsigned int Stage2LDSSize = 2 * (FFTSize2 + FFTSize2 / BANKS);
    constexpr unsigned int TransposeLDSSize = BlockSize * (FFTSize1 + PAD_T);
    constexpr unsigned int SingleBufferSize = Stage2LDSSize / 2;
    constexpr unsigned int LDS_Size =
        2 * Stage2LDSSize > TransposeLDSSize
        ? 2 * Stage2LDSSize
        : TransposeLDSSize;
    CUALGO_SHMEM float2 sdata[LDS_Size];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    // Pointer shift to correct batch
    T* idata1 = input_data1 + batch_id * IDataSize1;
    T* idata2 = input_data2 + batch_id * IDataSize2;
    T* odata = output_data + batch_id * FFTSize;

    float2 a[FFTSize1][FFTSize2 / BlockSize];
    float2 b[FFTSize1][FFTSize2 / BlockSize];

    // -------------------------------
    // STEP 1-2: FFT on cols + Twiddle
    // -------------------------------
    static_for<0, FFTSize2 / BlockSize>([&](auto i){
        stage1_static_columns<
            i.value,         // stage index
            FFTSize1,
            FFTSize2,
            BlockSize,
            IDataSize1,
            FFTSize1,        // NumCols = FFTSize1
            FFTSize2 / BlockSize  // NumRows
        >(idata1, idata2, a, b);          // pass the array directly, NOT &a[0][0]
    });

    float* buf0 = (float*)sdata;
    float* buf1 = (float*)(sdata + SingleBufferSize);
    float* buf2 = (float*)(sdata + 2 * SingleBufferSize);
    float* buf3 = (float*)(sdata + 3 * SingleBufferSize);

    // -------------------------------
    // STEP 3: FFT on rows
    // -------------------------------
    static_for<0, FFTSize1>([&](auto i){
        stage2_static_row<
            i.value,
            FFTSize1,
            FFTSize2,
            BlockSize,
            SingleBufferSize,
            FFTSize1,            // NumCols
            FFTSize2 / BlockSize // NumRows
        >(buf0, buf1, buf2, buf3, a, b);
    });

    int col = tid & (FFTSize1 - 1);
    constexpr int LOG2FFTSize1 = __builtin_ctz(FFTSize1);
    int row = tid >> LOG2FFTSize1;

    float2 c[FFTSize1][FFTSize2 / BlockSize];

    static_for<0, FFTSize2 / BlockSize>([&](auto j){
        __syncthreads();
        static_for<0, FFTSize1>([&](auto i){
            sdata[pad_transpose<FFTSize1>(tid,i.value)] =
                cmul(a[i.value][j.value], b[i.value][j.value]);
        });
        __syncthreads();

        static_for<0, FFTSize1>([&](auto i){
            // odata[tid + i.value * (BlockSize) + j.value * FFTSize1 * BlockSize] =
            // constexpr int reg_col = i.value * BlockSize / FFTSize2 +
            //                         j.value * FFTSize1 * BlockSize / FFTSize2;
            // constexpr int reg_row = i.value % (FFTSize2 / BlockSize);
            constexpr int reg_col = j.value;
            constexpr int reg_row = i.value;
            c[reg_col][reg_row] = 
                sdata[pad_transpose<FFTSize1>(row, col) +
                                    i.value * (FFTSize1 + PAD_T) * (BlockSize / FFTSize1)];
        });
    });

    // -------------------------------
    // STEP 1-2: IFFT on cols + Twiddle
    // -------------------------------
    static_for<0, FFTSize2 / BlockSize>([&](auto i){
        stage3_static_columns<
            i.value,         // stage index
            FFTSize1,
            FFTSize2,
            BlockSize,
            FFTSize1,        // NumCols = FFTSize1
            FFTSize2 / BlockSize  // NumRows
        >(c);          // pass the array directly, NOT &a[0][0]
    });

    buf0 = (float*)sdata;
    buf1 = (float*)(sdata + SingleBufferSize);

    __syncthreads();

    // -------------------------------
    // STEP 3: IFFT on rows
    // -------------------------------
    static_for<0, FFTSize1>([&](auto i){
        stage4_static_row<
            i.value,
            FFTSize1,
            FFTSize2,
            BlockSize,
            SingleBufferSize,
            FFTSize1,            // NumCols
            FFTSize2 / BlockSize // NumRows
        >(buf0, buf1, c);
    });

    static_for<0, FFTSize2 / BlockSize>([&](auto j){
        __syncthreads();
        static_for<0, FFTSize1>([&](auto i){
            sdata[pad_transpose<FFTSize1>(tid,i.value)] = make(c[i.value][j.value].x / FFTSize,
                                                               c[i.value][j.value].y / FFTSize);
        });
        __syncthreads();

        static_for<0, FFTSize1>([&](auto i){
            odata[tid + i.value * (BlockSize) + j.value * FFTSize1 * BlockSize] =
                sdata[pad_transpose<FFTSize1>(row, col) +
                                    i.value * (FFTSize1 + PAD_T) * (BlockSize / FFTSize1)];
        });
    });
}

namespace cuAlgo {

    /**
    * @brief   Perform 1d C2C fft
    * 
    * @details The input vector has dimension {size} and 
    *          the output vector has dimension {size}
    * 
    * @param[in]  idata pointer to input vector (complex)
    * @param[out] odata pointer to output vector (complex))
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <
    unsigned int FFTSize1,
    unsigned int FFTSize2,
    unsigned int IDataSize1,
    unsigned int IDataSize2,
    unsigned int BlockSize,
    typename T>
    void convFFT1dBailey(T *idata1,
                         T *idata2,
                         T *odata,
                         int batch_size,
                         cudaStream_t stream = 0,
                         bool async = false)
    {
        constexpr unsigned int FFTSize = FFTSize1 * FFTSize2;
        static_assert(is_power_of_two<FFTSize>());

        dim3 blocksPerGrid3(batch_size, 1, 1);
        dim3 threadsPerBlock3(BlockSize, 1, 1);
        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
            CUALGO_KERNEL_NAME(
                convFFT1dBaileyKernel<FFTSize1, FFTSize2, IDataSize1, IDataSize2, BlockSize, T>),
            idata1, idata2, odata, batch_size);
    }
}

#endif
