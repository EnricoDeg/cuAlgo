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
    int mh = 1;
    int t = N / 4;

    for (int s = 0; s < log4N; ++s) {

        for (int base = tid; base < N >> 2; base += BlockSize)
        {
            int i = base;
            int k = i % mh;

            float2 tw = twiddle(k, 4 * mh);
            // int twiddle_idx = (k * N) / (4 * mh);
            // float2 tw = HandleType::twiddles()[twiddle_idx];

            float2 a0 = in[i];
            float2 a1 = cmul(tw, in[i + t]);
            float2 a2 = in[i + 2 * t];
            float2 a3 = cmul(tw, in[i + 3 * t]);
            tw = cmul(tw, tw);
            a2 = cmul(tw, a2);
            a3 = cmul(tw, a3);

            float2 b0 = cadd(a0, a2);
            float2 b1 = csub(a0, a2);
            float2 b2 = cadd(a1, a3);
            float2 b3 = mul_neg_j(csub(a1, a3));

            int o = (i / mh) * mh * 4 + (i % mh);
            out[o + 0] = cadd(b0, b2);
            out[o + 1 * mh] = cadd(b1, b3);
            out[o + 2 * mh] = csub(b0, b2);
            out[o + 3 * mh] = csub(b1, b3);
        }

        __syncthreads();
        T* tmp = in; in = out; out = tmp;

        mh *= 4;
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
    }

    // ---- Store R elements per thread ----
    for (int base = tid; base < N; base += BlockSize) {
        odata[base] = in[base];
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
