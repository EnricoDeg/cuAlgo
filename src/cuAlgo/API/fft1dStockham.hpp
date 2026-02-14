/*
 * @file fft1dStockham.hpp
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

#ifndef FFT1DSTOCKHAM_HPP
#define FFT1DSTOCKHAM_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"
#include "cuAlgo/API/fft1dPlan.hpp"

template <
unsigned int N,
unsigned int BlockSize,
typename T,
typename HandleType>
__global__
void fft1dStockhamRadix2Kernel(T* __restrict__ input_data,
                               T* __restrict__ output_data,
                               int batch_size)
{
    __shared__ T smem[2 * N];

    T* idata = input_data  + blockIdx.x * N;
    T* odata = output_data + blockIdx.x * N;

    T* buf0 = smem;
    T* buf1 = smem + N;

    int tid = threadIdx.x; // 0 .. N/2-1

    // ---- Load 2 elements per thread ----
    int i = tid;
    buf0[i]       = idata[i];
    buf0[i + N/2] = idata[i + N/2];
    __syncthreads();

    T* in  = buf0;
    T* out = buf1;

    constexpr int log2N = __builtin_ctz(N);

    for (int s = 0; s < log2N; ++s) {
        int m  = 1 << (s + 1);
        int mh = m >> 1;

        int group = tid / mh;
        int j     = tid % mh;

        int i0 = group * mh + j;
        int i1 = i0 + N / 2;

        T a = in[i0];
        T b = in[i1];

        int tw = (j * N) / m;
        T w = HandleType::twiddles()[tw];
        T t = cmul(b, w);

        int o0 = group * m + j;
        int o1 = o0 + mh;

        out[o0] = cadd(a, t);
        out[o1] = csub(a, t);

        __syncthreads();
        T* tmp = in; in = out; out = tmp;
    }

    // ---- Store 2 elements per thread ----
    odata[i]       = in[i];
    odata[i + N/2] = in[i + N/2];
}

template <
unsigned int N,
unsigned int BlockSize,
typename T,
typename HandleType>
__global__
void fft1dStockhamMixedRadixKernel(T* __restrict__ input_data,
                                   T* __restrict__ output_data,
                                   int batch_size)
{
    __shared__ T smem[2 * N];

    T* idata = input_data  + blockIdx.x * N;
    T* odata = output_data + blockIdx.x * N;

    T* buf0 = smem;
    T* buf1 = smem + N;

    int tid = threadIdx.x;

    // ---- Load R elements per thread ----
    for (int base = tid; base < N; base += BlockSize) {
        buf0[base] = idata[base];
    }
    __syncthreads();

    T* in  = buf0;
    T* out = buf1;

    constexpr int log2N = __builtin_ctz(N);
    constexpr int log4N = log2N >> 1;
    constexpr bool isMixedRadix = (log2N & 1) != 0;
    constexpr int t = N / 4;
    constexpr int NIterations = isMixedRadix ? log4N : log4N - 1;

    for (int s = 0, mh = 1; s < NIterations; ++s, mh <<= 2)
    {
        for (int base = tid; base < N >> 2; base += BlockSize)
        {
            int i = base;
            int k = i & (mh - 1);

            // twiddle
            float2 tw = twiddle(k, 4 * mh);
            float wr = tw.x;
            float wi = tw.y;

            // load inputs
            float2 A0 = in[i];
            float2 A1 = in[i + t];
            float2 A2 = in[i + 2*t];
            float2 A3 = in[i + 3*t];

            // ---- a1 = tw * A1 ----
            float a1r = __fmaf_rn(-wi, A1.y, wr * A1.x);
            float a1i = __fmaf_rn( wr, A1.y, wi * A1.x);

            // ---- a3 = tw * A3 ----
            float a3r = __fmaf_rn(-wi, A3.y, wr * A3.x);
            float a3i = __fmaf_rn( wr, A3.y, wi * A3.x);

            // ---- tw = tw * tw  (square once) ----
            float wr2 = __fmaf_rn(-wi, wi, wr * wr);   // wr^2 - wi^2
            float wi2 = 2.f * wr * wi;                // 2wrwi

            // ---- a2 = tw^2 * A2 ----
            float a2r = __fmaf_rn(-wi2, A2.y, wr2 * A2.x);
            float a2i = __fmaf_rn( wr2, A2.y, wi2 * A2.x);

            // ---- a3 = tw^3 * A3  (we already did tw*A3, now multiply by tw^2) ----
            float tmp3r = __fmaf_rn(-wi2, a3i, wr2 * a3r);
            float tmp3i = __fmaf_rn( wr2, a3i, wi2 * a3r);
            a3r = tmp3r;
            a3i = tmp3i;

            // -------------------------------------
            // Radix-4 butterfly (fully fused)
            // -------------------------------------

            float b0r = A0.x + a2r;
            float b0i = A0.y + a2i;

            float b1r = A0.x - a2r;
            float b1i = A0.y - a2i;

            float b2r = a1r + a3r;
            float b2i = a1i + a3i;

            float d3r = a1r - a3r;
            float d3i = a1i - a3i;

            // mul_neg_j(x + iy) = ( y , -x )
            float b3r =  d3i;
            float b3i = -d3r;

            // output index (avoid division!)
            int o = (i - k) * 4 + k;

            // final outputs
            out[o + 0]      = make_float2(b0r + b2r, b0i + b2i);
            out[o + mh]     = make_float2(b1r + b3r, b1i + b3i);
            out[o + 2*mh]   = make_float2(b0r - b2r, b0i - b2i);
            out[o + 3*mh]   = make_float2(b1r - b3r, b1i - b3i);
        }

        __syncthreads();
        T* tmp = in; in = out; out = tmp;
    }

    if constexpr(isMixedRadix)
    {
        constexpr int half = N >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            T a = in[k];
            T b = in[k + N / 2];

            // Twiddle W_N^k
            T w = HandleType::twiddles()[k];

            T t = cmul(w, b);

            out[k]        = cadd(a, t);
            out[k + half] = csub(a, t);
        }

        __syncthreads();
        T* tmp = in; in = out; out = tmp;

        // ---- Store R elements per thread ----
        for (int base = tid; base < N; base += BlockSize) {
            odata[base] = in[base];
        }
    }
    else
    {
        // last stage + direct store
        int mh = t;
        for (int base = tid; base < N >> 2; base += BlockSize)
        {
            int i = base;
            int k = i & (mh - 1);

            // twiddle
            float2 tw = twiddle(k, 4 * mh);
            float wr = tw.x;
            float wi = tw.y;

            // load inputs
            float2 A0 = in[i];
            float2 A1 = in[i + t];
            float2 A2 = in[i + 2*t];
            float2 A3 = in[i + 3*t];

            // ---- a1 = tw * A1 ----
            float a1r = __fmaf_rn(-wi, A1.y, wr * A1.x);
            float a1i = __fmaf_rn( wr, A1.y, wi * A1.x);

            // ---- a3 = tw * A3 ----
            float a3r = __fmaf_rn(-wi, A3.y, wr * A3.x);
            float a3i = __fmaf_rn( wr, A3.y, wi * A3.x);

            // ---- tw = tw * tw  (square once) ----
            float wr2 = __fmaf_rn(-wi, wi, wr * wr);   // wr^2 - wi^2
            float wi2 = 2.f * wr * wi;                // 2wrwi

            // ---- a2 = tw^2 * A2 ----
            float a2r = __fmaf_rn(-wi2, A2.y, wr2 * A2.x);
            float a2i = __fmaf_rn( wr2, A2.y, wi2 * A2.x);

            // ---- a3 = tw^3 * A3  (we already did tw*A3, now multiply by tw^2) ----
            float tmp3r = __fmaf_rn(-wi2, a3i, wr2 * a3r);
            float tmp3i = __fmaf_rn( wr2, a3i, wi2 * a3r);
            a3r = tmp3r;
            a3i = tmp3i;

            // -------------------------------------
            // Radix-4 butterfly (fully fused)
            // -------------------------------------

            float b0r = A0.x + a2r;
            float b0i = A0.y + a2i;

            float b1r = A0.x - a2r;
            float b1i = A0.y - a2i;

            float b2r = a1r + a3r;
            float b2i = a1i + a3i;

            float d3r = a1r - a3r;
            float d3i = a1i - a3i;

            // mul_neg_j(x + iy) = ( y , -x )
            float b3r =  d3i;
            float b3i = -d3r;

            // output index (avoid division!)
            int o = (i - k) * 4 + k;

            // final outputs
            odata[o + 0]      = make_float2(b0r + b2r, b0i + b2i);
            odata[o + mh]     = make_float2(b1r + b3r, b1i + b3i);
            odata[o + 2*mh]   = make_float2(b0r - b2r, b0i - b2i);
            odata[o + 3*mh]   = make_float2(b1r - b3r, b1i - b3i);
        }
    }
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
    unsigned int FFTSize,
    unsigned int BlockSize,
    typename T>
    void fft1dStockham(T *idata ,
                       T *odata ,
                       int batch_size,
                       cudaStream_t stream = 0,
                       bool async = false)
    {
        dim3 blocksPerGrid3(batch_size, 1, 1);
        dim3 threadsPerBlock3(BlockSize, 1, 1);
        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
             CUALGO_KERNEL_NAME(
                fft1dStockhamMixedRadixKernel<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
             idata, odata, batch_size);
    }
}

#endif
