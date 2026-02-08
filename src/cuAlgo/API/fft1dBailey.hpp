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

__device__ inline float2 W(int N, int k){
    float ang = -2.f * M_PI * k / N;
    return make_float2(cosf(ang), sinf(ang));
}

template<unsigned int FFTSize>
__device__ void fft2_reg(float2 *a)
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
__device__ void fft4_reg(float2 *a)
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
__device__ void fft8_reg(float2 *a)
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

#define BANKS 32

__device__ __forceinline__
int pad(int i) {
    return i + (i >> 5);   // i / 32
}

template<
unsigned int FFTSize1,
unsigned int FFTSize2,
unsigned int BlockSize,
typename T,
typename HandleType>
__global__
void fft1dBaileyKernel(T* input_data,
                       T* output_data,
                       int batch_size)
{
    constexpr unsigned int FFTSize = FFTSize1 * FFTSize2;
    __shared__ float2 sdata[FFTSize2 + FFTSize2 / BANKS];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    // Pointer shift to correct batch
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    static_assert(FFTSize1 == 4 || FFTSize1 == 8 || FFTSize1 == 2);

    float2 a[FFTSize1 * (FFTSize2 / BlockSize)];
    for(int tidx = tid; tidx < FFTSize2; tidx += BlockSize)
    {
        int offset = tidx / BlockSize;
        // --------------------------
        // STEP 1: FFT on columns
        // --------------------------
        for(int i = 0; i < FFTSize1; ++i)
        {
            a[i + offset * FFTSize1] = idata[tidx + FFTSize2 * i];
        }

        if constexpr(FFTSize1 == 4)
        {
            fft4_reg<FFTSize1>(&a[offset * FFTSize1]);
        }
        else if constexpr(FFTSize1 == 8)
        {
            fft8_reg<FFTSize1>(&a[offset * FFTSize1]);
        }
        else if constexpr(FFTSize1 == 2)
        {
            fft2_reg<FFTSize1>(&a[offset * FFTSize1]);
        }

        // --------------------------
        // STEP 2: Multiply twiddle factors
        // W16^(n1 * n2)
        // Store back in strided layout
        // --------------------------
        for(int i = 0; i < FFTSize1; ++i)
        {
            a[i + offset * FFTSize1] =
                cmul(a[i + offset * FFTSize1], W(FFTSize2 * FFTSize1, tidx * i));
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
        for(int tidx = tid; tidx < FFTSize2; tidx += BlockSize)
        {
            int offset = tidx / BlockSize;
            unsigned int r = isMixedRadix ?
                             mixed_radix_reverse(tidx, log4N) :
                             base4_reverse(tidx, log4N);
            sdata[pad(r)] = a[n2 + offset * FFTSize1];
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
        for(int i = tid; i < FFTSize2; i += BlockSize)
        {
            odata[i * FFTSize1 + n2] = sdata[pad(i)];
            // odata[i + FFTSize2 * n2] = sdata[pad(i)];
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
                fft1dBaileyKernel<FFTSize1, FFTSize2, BlockSize, T, fftHandle<FFTSize2>>),
            idata, odata, batch_size);
    }
}

#endif
