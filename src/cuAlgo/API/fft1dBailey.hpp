/*
 * @file fft1dCT.hpp
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

#ifndef FFT1DBAILEY_HPP
#define FFT1DBAILEY_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"
#include "cuAlgo/API/fft1dPlan.hpp"


// -------------------------------------------------------------------------------------------------
// Naive
// -------------------------------------------------------------------------------------------------
#define BANKS 32

CUALGO_DEVICE CUALGO_FORCE_INLINE
int pad(int i) {
    return i + (i >> 5);   // i / 32
}

CUALGO_DEVICE CUALGO_FORCE_INLINE
float2 W(int N, int k){
    float ang = -2.f * M_PI * k / N;
    return make_float2(cosf(ang), sinf(ang));
}

template<unsigned int FFTSize>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fft2_reg(float2 *a)
{
    int len = 2;
    int half = len >> 1;
    int tidx = 0;
    int block = tidx / half;
    int k = tidx % half;
    int i = block * len + k;

    float2 w = make(1.0f, 0.0f);
    float2 u = a[i];
    float2 v = cmul(w, a[i + half]);

    a[i]       = cadd(u, v);
    a[i + half]= csub(u, v);
}

template<unsigned int FFTSize>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fft4_reg(float2 *a)
{
    constexpr int LOG2N = __builtin_ctz(FFTSize);
    constexpr int log4N = LOG2N >> 1;

    // bit reverse
    float2 tmp[FFTSize];
    for(int i = 0; i < FFTSize; ++i)
    {
        unsigned int r = base4_reverse(i, log4N);
        tmp[r] = a[i];
    }

    // Radix4 in register
    for (int stage = 0, m = 4; stage < log4N; ++stage, m <<= 2)
    {
        int quarter = m >> 2;

        for (int base = 0; base < FFTSize >> 2; ++base)
        {
            int j = base % quarter;
            int k = base / quarter;
            int p = k * m + j;

            float2 W1 = W(m, j);
            float2 W2 = cmul(W1, W1);
            float2 W3 = cmul(W2, W1);

            float2 x0 = tmp[p + 0 * quarter];
            float2 x1 = cmul(W1, tmp[p + 1 * quarter]);
            float2 x2 = cmul(W2, tmp[p + 2 * quarter]);
            float2 x3 = cmul(W3, tmp[p + 3 * quarter]);

            float2 t0 = cadd(x0, x2);
            float2 t1 = cadd(x1, x3);
            float2 t2 = csub(x0, x2);
            float2 t3 = csub(x1, x3);

            a[p + 0 * quarter] = cadd(t0, t1);
            a[p + 2 * quarter] = csub(t0, t1);
            a[p + 1 * quarter] = make(t2.x + t3.y, t2.y - t3.x);
            a[p + 3 * quarter] = make(t2.x - t3.y, t2.y + t3.x);
        }
    }
}

template<unsigned int FFTSize>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fft8_reg(float2 *a)
{
    constexpr int log8N = __builtin_ctz(FFTSize) / 3;

    // bit reverse
    float2 tmp[FFTSize];
    for(int i = 0; i < FFTSize; ++i)
    {
        unsigned int r = base8_reverse(i, log8N);
        tmp[r] = a[i];
    }

    // Radix8 in register
    for(int stage = 0, len = 8; stage < log8N; ++stage, len <<= 3) {

        int stride  = len >> 3;
        int butterflies = (FFTSize / len) * stride;

        for(int base = 0; base < FFTSize >> 3; ++base)
        {
            if (base < butterflies) {
                int group = base / stride;
                int k     = base % stride;
                int base  = group * len + k;
                // int tw    = k * tw_step;

                // radix8_butterfly<HandleType>(&sdata[base], stride, tw, FFTSize);
                float2 a0 = tmp[0];
                float2 a1 = cmul(tmp[base + stride],     W(len, 1 * k)); //HandleType::twiddles()[tw * 1]);
                float2 a2 = cmul(tmp[base + 2 * stride], W(len, 2 * k)); //HandleType::twiddles()[tw * 2]);
                float2 a3 = cmul(tmp[base + 3 * stride], W(len, 3 * k)); //HandleType::twiddles()[tw * 3]);
                float2 a4 = cmul(tmp[base + 4 * stride], W(len, 4 * k)); //HandleType::twiddles()[tw * 4]);
                float2 a5 = cmul(tmp[base + 5 * stride], W(len, 5 * k)); //HandleType::twiddles()[tw * 5]);
                float2 a6 = cmul(tmp[base + 6 * stride], W(len, 6 * k)); //HandleType::twiddles()[tw * 6]);
                float2 a7 = cmul(tmp[base + 7 * stride], W(len, 7 * k)); //HandleType::twiddles()[tw * 7]);

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

                a[base + 0]           = cadd(t0, t1);
                a[base + stride]      = cadd(u0, mul_W8_1(u1));
                a[base + 2 * stride]  = cadd(t2, t3);
                a[base + 3 * stride]  = cadd(u2, mul_W8_3(u3));
                a[base + 4 * stride]  = csub(t0, t1);
                a[base + 5 * stride]  = csub(u0, mul_W8_1(u1));
                a[base + 6 * stride]  = csub(t2, t3);
                a[base + 7 * stride]  = csub(u2, mul_W8_3(u3));

            }
        }
    }
}

template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
typename T,
typename HandleType>
__global__ __launch_bounds__(BlockSize)
void fft1dBaileyKernelNaive(T* input_data,
                            T* output_data,
                            int batch_size)
{
    constexpr unsigned int FFTSize = FFTSize1 * FFTSize2;
    constexpr unsigned int LDS_Size =
        FFTSize2 + FFTSize2 / BANKS > BlockSize * FFTSize1
        ? FFTSize2 + FFTSize2 / BANKS
        : BlockSize * FFTSize1;
    __shared__ float2 sdata[LDS_Size];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    // Pointer shift to correct batch
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    static_assert(FFTSize1 == 4 || FFTSize1 == 8 || FFTSize1 == 2);

    float2 a[FFTSize1 * (FFTSize2 / BlockSize)];
    #pragma unroll
    for(int n = 0; n < FFTSize2 / BlockSize; ++n)
    {
        // --------------------------
        // STEP 1: FFT on columns
        // --------------------------
        #pragma unroll
        for(int i = 0; i < FFTSize1; ++i)
        {
            a[i + n * FFTSize1] = idata[tid + n * BlockSize + FFTSize2 * i];
        }

        if constexpr(FFTSize1 == 4)
        {
            fft4_reg<FFTSize1>(&a[n * FFTSize1]);
        }
        else if constexpr(FFTSize1 == 8)
        {
            fft8_reg<FFTSize1>(&a[n * FFTSize1]);
        }
        else if constexpr(FFTSize1 == 2)
        {
            fft2_reg<FFTSize1>(&a[n * FFTSize1]);
        }

        // --------------------------
        // STEP 2: Multiply twiddle factors
        // W16^(n1 * n2)
        // Store back in strided layout
        // --------------------------
        #pragma unroll
        for(int i = 0; i < FFTSize1; ++i)
        {
            a[i + n * FFTSize1] =
                cmul(a[i + n * FFTSize1],
                     W(FFTSize2 * FFTSize1, (tid + n * BlockSize) * i));
        }
    }

    // --------------------------
    // STEP 3: FFT on rows
    // --------------------------
    for(int n2 = 0; n2 < FFTSize1; n2++)
    {
        constexpr int LOG2N = __builtin_ctz(FFTSize2);
        constexpr bool isMixedRadix = (LOG2N & 1) != 0;
        constexpr int log4N = LOG2N >> 1;

        // Load one row into shared memory
        #pragma unroll
        for(int i = 0; i < FFTSize2 / BlockSize; ++i)
        {
            unsigned int r = isMixedRadix ?
                             mixed_radix_reverse(tid + i * BlockSize, log4N) :
                             base4_reverse(tid + i * BlockSize, log4N);
            sdata[pad(r)] = a[n2 + i * FFTSize1];
        }
        __syncthreads();

        for (int stage = 0, m = 4; stage < log4N; ++stage, m <<= 2)
        {
            int quarter = m >> 2;

            for (int base = tid; base < FFTSize2 >> 2; base += BlockSize)
            {
                int j = base % quarter;
                int k = base / quarter;
                int p = k * m + j;

                float2 W1 = W(m, j);
                float2 W2 = cmul(W1, W1);
                float2 W3 = cmul(W2, W1);

                float2 x0 = sdata[pad(p + 0 * quarter)];
                float2 x1 = cmul(W1, sdata[pad(p + 1 * quarter)]);
                float2 x2 = cmul(W2, sdata[pad(p + 2 * quarter)]);
                float2 x3 = cmul(W3, sdata[pad(p + 3 * quarter)]);

                float2 t0 = cadd(x0, x2);
                float2 t1 = cadd(x1, x3);
                float2 t2 = csub(x0, x2);
                float2 t3 = csub(x1, x3);

                sdata[pad(p + 0 * quarter)] = cadd(t0, t1);
                sdata[pad(p + 2 * quarter)] = csub(t0, t1);
                sdata[pad(p + 1 * quarter)] = make(t2.x + t3.y, t2.y - t3.x);
                sdata[pad(p + 3 * quarter)] = make(t2.x - t3.y, t2.y + t3.x);
            }
            __syncthreads();
        }

        if constexpr(isMixedRadix)
        {

            constexpr int half = FFTSize2 >> 1;

            for (int k = tid; k < half; k += BlockSize) {

                T a = sdata[pad(k       )];
                T b = sdata[pad(k + half)];

                // Twiddle W_N^k
                T w = W(FFTSize2, k);

                T t = cmul(w, b);

                sdata[pad(k       )] = cadd(a, t);
                sdata[pad(k + half)] = csub(a, t);
            }

            __syncthreads();
        }

        // Store back to global memory
        #pragma unroll
        for(int i = 0; i < FFTSize2 / BlockSize; ++i)
        {
            a[n2 + i * FFTSize1] = sdata[pad(i * BlockSize + tid)];
        }

        __syncthreads();
    }

    #pragma unroll
    for(int n = 0; n < FFTSize2 / BlockSize; ++n)
    {
        __syncthreads();
        #pragma unroll
        for(int n2 = 0; n2 < FFTSize1; n2++)
        {
            sdata[n2 + tid * FFTSize1] = a[n2 + n * FFTSize1];
        }
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < FFTSize1; ++i)
        {
            odata[tid + i * BlockSize + n * FFTSize1 * BlockSize] = sdata[i * BlockSize + tid];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Experiment 32x32 warp shuffling
// -------------------------------------------------------------------------------------------------
template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
typename T,
typename HandleType>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void fft1dBailey32x32Impl(T* sdata)
{
    constexpr int LOG2N = __builtin_ctz(FFTSize1);

    int tid = threadIdx.x;
    unsigned int lane = tid & (31);   // lane id in warp
    int warp_id = tid >> 5; // warp index

    static_assert(FFTSize1 == 32);
    static_assert(FFTSize2 == 32);

    unsigned mask = 0xffffffff;

    constexpr int NWarps = BlockSize / 32;

    __syncthreads();

    unsigned int row_idx_reversed = base2_reverse(lane, LOG2N);

    float2 twiddles[LOG2N];
    for(int stage = 0, len = 2; stage < LOG2N; ++stage, len <<= 1)
    {
        int half = len >> 1;
        int k = lane & (half - 1);
        int twiddle_idx = (k * FFTSize1) / len;
        twiddles[stage] = HandleType::twiddles()[twiddle_idx];
    }

    // FFT stages
    #pragma unroll
    for(int n = 0; n < FFTSize2 / NWarps; ++n)
    {
        // load transpose + bit-reversed
        float2 x = sdata[row_idx_reversed * (FFTSize2 + 1) + warp_id + n * NWarps];

        #pragma unroll
        for(int stage = 0, len = 2; stage < LOG2N; ++stage, len <<= 1)
        {
            int half = len >> 1;

            float2 w = twiddles[stage];

            float2 y;
            y.x = __shfl_xor_sync(mask, x.x, half);
            y.y = __shfl_xor_sync(mask, x.y, half);

            if ((lane & (len - 1)) < half)
            {
                float xr = x.x;
                float xi = x.y;
                float yr = y.x;
                float yi = y.y;
                float wr = w.x;
                float wi = w.y;

                // x = x + w * y
                x.x = fmaf(wr, yr, xr) - wi * yi;
                x.y = fmaf(wr, yi, xi) + wi * yr;
            }
            else
            {
                float xr = x.x;
                float xi = x.y;
                float yr = y.x;
                float yi = y.y;
                float wr = w.x;
                float wi = w.y;

                // x = y - w * x
                float tr = fmaf(wr, xr, -wi * xi);
                float ti = fmaf(wr, xi,  wi * xr);

                x.x = yr - tr;
                x.y = yi - ti;
            }
        }

        // __syncwarp();

        x = cmul(x, W(FFTSize2 * FFTSize1, (warp_id + n * NWarps) * lane));
        sdata[lane * (FFTSize2 + 1) + warp_id + n * NWarps] = x;
    }

    __syncthreads();

    #pragma unroll
    for(int n = 0; n < FFTSize1 / NWarps; ++n)
    {
        // load transpose + bit-reversed
        float2 x = sdata[row_idx_reversed + (warp_id + n * NWarps) * (FFTSize2 + 1)];

        #pragma unroll
        for(int stage = 0, len = 2; stage < LOG2N; ++stage, len <<= 1)
        {
            int half = len >> 1;

            float2 w = twiddles[stage];

            float2 y;
            y.x = __shfl_xor_sync(mask, x.x, half);
            y.y = __shfl_xor_sync(mask, x.y, half);

            if ((lane & (len - 1)) < half)
            {
                // float2 t = cmul(w, y);
                // x = cadd(x, t);
                float xr = x.x;
                float xi = x.y;
                float yr = y.x;
                float yi = y.y;
                float wr = w.x;
                float wi = w.y;

                // x = x + w * y
                x.x = fmaf(wr, yr, xr) - wi * yi;
                x.y = fmaf(wr, yi, xi) + wi * yr;
            }
            else
            {
                // float2 t = cmul(w, x);
                // x = csub(y, t);
                float xr = x.x;
                float xi = x.y;
                float yr = y.x;
                float yi = y.y;
                float wr = w.x;
                float wi = w.y;

                // x = y - w * x
                float tr = fmaf(wr, xr, -wi * xi);
                float ti = fmaf(wr, xi,  wi * xr);

                x.x = yr - tr;
                x.y = yi - ti;
            }
        }

        // __syncwarp();

        sdata[lane + (warp_id + n * NWarps) * (FFTSize2 + 1)] = x;
    }

    __syncthreads();
}

template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
typename T,
typename HandleType>
CUALGO_GLOBAL CUALGO_LAUNCH_BOUNDS(BlockSize)
void fft1dBailey32x32Kernel(T* input_data,
                            T* output_data,
                            int batch_size)
{
    constexpr unsigned int FFTSize = FFTSize1 * FFTSize2;
    constexpr unsigned int LDS_Size = FFTSize1 * (FFTSize2 + 1);
    constexpr int LOG2N = __builtin_ctz(FFTSize1);
    __shared__ float2 sdata[LDS_Size];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    unsigned int lane = tid & (31);   // lane id in warp
    int warp_id = tid >> 5; // warp index

    // Pointer shift to correct batch
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    for(int i = tid; i < FFTSize; i += BlockSize)
    {
        int col = i & (FFTSize2 - 1);
        int row = i / FFTSize2;
        sdata[col + row * (FFTSize2 + 1)] = idata[col + row * FFTSize2];
    }

    fft1dBailey32x32Impl<FFTSize1, FFTSize2, BlockSize, T, HandleType>(sdata);

    constexpr int NWarps = BlockSize / 32;

    #pragma unroll
    for(int n = 0; n < FFTSize2 / NWarps; ++n)
    {
        odata[lane + (warp_id + n * NWarps) * FFTSize1] =
            sdata[lane * (FFTSize2 + 1) + warp_id + n * NWarps];
    }
}

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
void fftN_reg_fast(float2 *a0, float2 *a1, float2 *a2, float2 *a3)
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

    *a1 = make(t0.x + t1.y, t0.y - t1.x);
    *a3 = make(t0.x - t1.y, t0.y + t1.x);
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
    typename... Columns>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage1(float2* CUALGO_RESTRICT idata, Columns* CUALGO_RESTRICT... columns)
{
    int tid = threadIdx.x;
    constexpr int N = sizeof...(Columns);

    // Turn parameter pack into an array of pointers
    float2* ptrs[N] = { columns... };

    // Compile-time unrolled assignment using static_for
    static_for<0, N>([&](auto I){
        *ptrs[I.value] = *(idata + tid + It * BlockSize + FFTSize2 * I.value);
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

        float inv =  GET_INV(0); // 1.f / (mh << 2);
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

        out[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
        out[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
        out[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
        out[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);

        __syncthreads();
        float* tmp = in; in = out; out = tmp;
    }

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

        out[permute(o + 1*mh)] = __fmaf_rn(1.f, b0_real,  b2_real);
        out[permute(o + 1*mh) + Offset] = __fmaf_rn(1.f, b0_imag,  b2_imag);
        out[permute(o + 3*mh)] = __fmaf_rn(1.f, b0_real, -b2_real);
        out[permute(o + 3*mh) + Offset] = __fmaf_rn(1.f, b0_imag, -b2_imag);

        __syncthreads();
        float* tmp = in; in = out; out = tmp;
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

        (*ptrs[1]).x = __fmaf_rn(1.f, b0_real,  b2_real);
        (*ptrs[1]).y = __fmaf_rn(1.f, b0_imag,  b2_imag);
        (*ptrs[3]).x = __fmaf_rn(1.f, b0_real, -b2_real);
        (*ptrs[3]).y = __fmaf_rn(1.f, b0_imag, -b2_imag);
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
    int NumCols,
    int NumRows
>
CUALGO_DEVICE CUALGO_FORCE_INLINE
void stage1_static_columns(float2* idata, float2 (&a)[NumCols][NumRows])
{
    float2* cols[NumCols];
    get_column_ptrs<NumCols, NumRows, It>(cols, a);

    expand_array(cols, [&](auto... args){
        stage1<It, FFTSize1, FFTSize2, BlockSize>(idata, args...);
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
                       float2 (&a)[NumCols][NumRows])
{
    float2* rows[NumRows];
    get_row_ptrs<NumCols, NumRows, It>(rows, a);

    // Expand array into variadic stage2
    expand_array(rows, [&](auto... args){
        stage2<It, FFTSize1, FFTSize2, BlockSize, Offset>(buf0, buf1, args...);
    }, std::make_integer_sequence<int, NumRows>{});
}

template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
typename T>
__global__ __launch_bounds__(BlockSize)
void fft1dBaileyKernel(T* CUALGO_RESTRICT input_data,
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
        Stage2LDSSize > TransposeLDSSize
        ? Stage2LDSSize
        : TransposeLDSSize;
    CUALGO_SHMEM float2 sdata[LDS_Size];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    // Pointer shift to correct batch
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    float2 a[FFTSize1][FFTSize2 / BlockSize];

    // -------------------------------
    // STEP 1-2: FFT on cols + Twiddle
    // -------------------------------
    static_for<0, FFTSize2 / BlockSize>([&](auto i){
        stage1_static_columns<
            i.value,         // stage index
            FFTSize1,
            FFTSize2,
            BlockSize,
            FFTSize1,        // NumCols = FFTSize1
            FFTSize2 / BlockSize  // NumRows
        >(idata, a);          // pass the array directly, NOT &a[0][0]
    });

    float* buf0 = (float*)sdata;
    float* buf1 = (float*)(sdata + SingleBufferSize);

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
        >(buf0, buf1, a);
    });

    // -------------------------------
    // STEP 4: final transpose
    // -------------------------------
    int col = tid & (FFTSize1 - 1);
    constexpr int LOG2FFTSize1 = __builtin_ctz(FFTSize1);
    int row = tid >> LOG2FFTSize1;

    static_for<0, FFTSize2 / BlockSize>([&](auto j){
        __syncthreads();
        static_for<0, FFTSize1>([&](auto i){
            sdata[pad_transpose<FFTSize1>(tid,i.value)] = a[i.value][j.value];
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
    unsigned int BlockSize,
    typename T>
    void fft1dBailey(T *idata,
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
                fft1dBaileyKernel<FFTSize1, FFTSize2, BlockSize, T>),
            idata, odata, batch_size);
    }
}

#endif
