/*
 * @file convfft1dCT.hpp
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

#ifndef CONVFFT1DCT_HPP
#define CONVFFT1DCT_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"
#include "cuAlgo/API/fft1dPlan.hpp"

constexpr int STATIC_SMEM_LIMIT = 48 * 1024;

template<int Bytes, bool UseStatic = (Bytes <= STATIC_SMEM_LIMIT)>
struct SharedMemory;

template<int Bytes>
struct SharedMemory<Bytes, true>
{
    __device__ static float2* get()
    {
        __shared__ float2 smem[Bytes / sizeof(float2)];
        return smem;
    }
};

template<int Bytes>
struct SharedMemory<Bytes, false>
{
    __device__ static float2* get()
    {
        extern __shared__ float2 smem[];
        return smem;
    }
};

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleTypeFwd,
typename HandleTypeBwd
>
CUALGO_GLOBAL
void convFFT1dCTKernelRadix2DITDIT(T * CUALGO_RESTRICT input_data1,
                                   T * CUALGO_RESTRICT input_data2,
                                   T * CUALGO_RESTRICT output_data,
                                   int input1_size,
                                   int input2_size,
                                   int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    float2* sdata = SharedMemory<FFTSize * 2 * sizeof(T)>::get();

    float2 *sdata1 = sdata;
    float2 *sdata2 = sdata + halfN;

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    float* idata1 = input_data1 + batch_id * input1_size;
    float* idata2 = input_data2 + batch_id * input2_size;
    float* odata  = output_data + batch_id * FFTSize;
    const int LOGN = __ffs(halfN) - 1;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store + Padding
    // ------------------------------------------------
    for (int idx = tid; idx < input1_size / 2; idx += BlockSize)
    {
        unsigned int r = base2_reverse(idx, LOGN);
        float2 val = make(idata1[2 * idx], idata1[2 * idx + 1]);
        sdata1[r] = val;
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        unsigned int r = base2_reverse(idx, LOGN);
        sdata1[r] = make(0.0f, 0.0f);
    }

    for (int idx = tid; idx < input2_size / 2; idx += BlockSize)
    {
        unsigned int r = base2_reverse(idx, LOGN);
        float2 val = make(idata2[2 * idx], idata2[2 * idx + 1]);
        sdata2[r] = val;
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        unsigned int r = base2_reverse(idx, LOGN);
        sdata2[r] = make(0.0f, 0.0f);
    }

    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages FFT
    // ------------------------------------------------
    radix2_CT_DIT<halfN, BlockSize, float2, HandleTypeFwd>(sdata1, tid, LOGN);
    radix2_CT_DIT<halfN, BlockSize, float2, HandleTypeFwd>(sdata2, tid, LOGN);

    // ------------------------------------------------
    // 3. Post-processing FFT + Convolution
    // ------------------------------------------------

    // Post-process FFT and compute convolution (results in registers)
    float2 res[(halfN + 1) / BlockSize + 1];
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        float ang          = -2.0f * M_PI * idx / FFTSize;
        float2 W           = make(cosf(ang), sinf(ang));
        float2 res1, res2;
        // First signal post-process
        {
            float2 Zk          = sdata1[idx % halfN];
            float2 Zk_conj     = conjf2(sdata1[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res1               = cadd(E, cmul(W, O));
        }

        // Second signal post-process
        {
            float2 Zk          = sdata2[idx % halfN];
            float2 Zk_conj     = conjf2(sdata2[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res2               = cadd(E, cmul(W, O));
        }

        res[idx / BlockSize] = cmul(res1, res2);
    }

    __syncthreads();

    // Move results to smem
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        sdata[idx] = res[idx/BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 4. Pre-processing iFFT
    // ------------------------------------------------

    // Pre-process inverse FFT and store results in register
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 Zk;
        float2 Ck      = sdata[idx];
        float2 Cmir    = conjf2(sdata[(halfN - idx)]);
        // Compute E and O
        float2 E = make(0.5f * (Ck.x + Cmir.x), 0.5f * (Ck.y + Cmir.y));
        float2 O = make(0.5f * (Ck.x - Cmir.x), 0.5f * (Ck.y - Cmir.y));
        // Twiddle
        float ang = 2.0f * M_PI * idx / FFTSize;
        float2 W  = make(cosf(ang), sinf(ang));
        float2 W1 = make(-W.y, W.x);
        Zk = cadd(E,cmul(W1,O));
        // Store Zk in register
        res[idx / BlockSize] = Zk;
    }

    __syncthreads();

    // Move results to smem and bit reverse
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        // bit reverse and store in smem
        unsigned int r = base2_reverse(idx, LOGN);
        sdata[r] = res[idx / BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 5. radix-2 stages inverse FFT
    // ------------------------------------------------
    radix2_CT_DIT<halfN, BlockSize, float2, HandleTypeBwd>(sdata, tid, LOGN);

    // ------------------------------------------------
    // 6. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize) {
        odata[2*idx]     = sdata[idx].x / (FFTSize / 2);  // even samples
        odata[2*idx + 1] = sdata[idx].y / (FFTSize / 2);  // odd  samples
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleTypeFwd,
typename HandleTypeBwd
>
CUALGO_GLOBAL
void convFFT1dCTKernelMixedRadixDITDIT(T * CUALGO_RESTRICT input_data1,
                                       T * CUALGO_RESTRICT input_data2,
                                       T * CUALGO_RESTRICT output_data,
                                       int input1_size,
                                       int input2_size,
                                       int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    float2* sdata = SharedMemory<FFTSize * 2 * sizeof(T)>::get();

    float2 *sdata1 = sdata;
    float2 *sdata2 = sdata + halfN;

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    float* idata1 = input_data1 + batch_id * input1_size;
    float* idata2 = input_data2 + batch_id * input2_size;
    float* odata  = output_data + batch_id * FFTSize;
    const int LOG2N = __builtin_ctz(halfN);
    constexpr int log4N = LOG2N >> 1;
    constexpr bool isMixedRadix = (LOG2N & 1) != 0;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store + Padding
    // ------------------------------------------------
    for (int idx = tid; idx < input1_size / 4; idx += BlockSize)
    {
        float4 val4 = *(float4 *)(idata1 + 4 * idx);
        {
            float2 val = make(val4.x, val4.y);
            unsigned int r = isMixedRadix ?
                            mixed_radix_reverse(2 * idx, log4N) :
                            base4_reverse(2 * idx, log4N);
            sdata1[r] = val;
        }
        {
            float2 val = make(val4.z, val4.w);
            unsigned int r = isMixedRadix ?
                             mixed_radix_reverse(2 * idx + 1, log4N) :
                             base4_reverse(2 * idx + 1, log4N);
            sdata1[r] = val;
        }
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        unsigned int r = isMixedRadix ?
                         mixed_radix_reverse(idx, log4N) :
                         base4_reverse(idx, log4N);
        sdata1[r] = make(0.0f, 0.0f);
    }

    for (int idx = tid; idx < input2_size / 4; idx += BlockSize)
    {
        float4 val4 = *(float4 *)(idata2 + 4 * idx);
        {
            float2 val = make(val4.x, val4.y);
            unsigned int r = isMixedRadix ?
                            mixed_radix_reverse(2 * idx, log4N) :
                            base4_reverse(2 * idx, log4N);
            sdata2[r] = val;
        }
        {
            float2 val = make(val4.z, val4.w);
            unsigned int r = isMixedRadix ?
                             mixed_radix_reverse(2 * idx + 1, log4N) :
                             base4_reverse(2 * idx + 1, log4N);
            sdata2[r] = val;
        }
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        unsigned int r = isMixedRadix ?
                         mixed_radix_reverse(idx, log4N) :
                         base4_reverse(idx, log4N);
        sdata2[r] = make(0.0f, 0.0f);
    }

    __syncthreads();

    // ------------------------------------------------
    // 2a. radix-4 stages FFT first signal
    // ------------------------------------------------
    for (int stage = 0, m = 4; stage < log4N; ++stage, m <<= 2)
    {

        int quarter = m >> 2;

        for (int base = tid; base < halfN >> 2; base += BlockSize)
        {
            int j = base % quarter;
            int k = base / quarter;
            int p = k * m + j;

            int twiddle_idx = (j * halfN) / m;
            float2 W1 = HandleTypeFwd::twiddles()[twiddle_idx];
            float2 W2 = cmul(W1, W1);
            float2 W3 = cmul(W2, W1);

            float2 x0_1 = sdata1[p + 0 * quarter];
            float2 x1_1 = cmul(W1, sdata1[p + 1 * quarter]);
            float2 x2_1 = cmul(W2, sdata1[p + 2 * quarter]);
            float2 x3_1 = cmul(W3, sdata1[p + 3 * quarter]);

            float2 x0_2 = sdata2[p + 0 * quarter];
            float2 x1_2 = cmul(W1, sdata2[p + 1 * quarter]);
            float2 x2_2 = cmul(W2, sdata2[p + 2 * quarter]);
            float2 x3_2 = cmul(W3, sdata2[p + 3 * quarter]);

            float2 t0_1 = cadd(x0_1, x2_1);
            float2 t1_1 = cadd(x1_1, x3_1);
            float2 t2_1 = csub(x0_1, x2_1);
            float2 t3_1 = csub(x1_1, x3_1);

            float2 t0_2 = cadd(x0_2, x2_2);
            float2 t1_2 = cadd(x1_2, x3_2);
            float2 t2_2 = csub(x0_2, x2_2);
            float2 t3_2 = csub(x1_2, x3_2);

            sdata1[p + 0 * quarter] = cadd(t0_1, t1_1);
            sdata1[p + 2 * quarter] = csub(t0_1, t1_1);
            sdata1[p + 1 * quarter] = make(t2_1.x + t3_1.y, t2_1.y - t3_1.x);
            sdata1[p + 3 * quarter] = make(t2_1.x - t3_1.y, t2_1.y + t3_1.x);

            sdata2[p + 0 * quarter] = cadd(t0_2, t1_2);
            sdata2[p + 2 * quarter] = csub(t0_2, t1_2);
            sdata2[p + 1 * quarter] = make(t2_2.x + t3_2.y, t2_2.y - t3_2.x);
            sdata2[p + 3 * quarter] = make(t2_2.x - t3_2.y, t2_2.y + t3_2.x);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 2b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix) {

        constexpr int half = halfN >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            float2 a1 = sdata1[k];
            float2 b1 = sdata1[k + half];

            float2 a2 = sdata2[k];
            float2 b2 = sdata2[k + half];

            // Twiddle W_N^k
            float2 w = HandleTypeFwd::twiddles()[k];

            float2 t1 = cmul(w, b1);
            float2 t2 = cmul(w, b2);

            sdata1[k]        = cadd(a1, t1);
            sdata1[k + half] = csub(a1, t1);

            sdata2[k]        = cadd(a2, t2);
            sdata2[k + half] = csub(a2, t2);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Post-processing FFT + Convolution
    // ------------------------------------------------

    // Post-process FFT and compute convolution (results in registers)
    float2 res[(halfN + 1) / BlockSize + 1];
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        float ang          = -2.0f * M_PI * idx / FFTSize;
        float2 W           = make(cosf(ang), sinf(ang));
        float2 res1, res2;
        // First signal post-process
        {
            float2 Zk          = sdata1[idx % halfN];
            float2 Zk_conj     = conjf2(sdata1[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res1               = cadd(E, cmul(W, O));
        }

        // Second signal post-process
        {
            float2 Zk          = sdata2[idx % halfN];
            float2 Zk_conj     = conjf2(sdata2[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res2               = cadd(E, cmul(W, O));
        }

        res[idx / BlockSize] = cmul(res1, res2);
    }

    __syncthreads();

    // Move results to smem
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        sdata[idx] = res[idx/BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 4. Pre-processing iFFT
    // ------------------------------------------------

    // Pre-process inverse FFT and store results in register
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 Zk;
        float2 Ck      = sdata[idx];
        float2 Cmir    = conjf2(sdata[(halfN - idx)]);
        // Compute E and O
        float2 E = make(0.5f * (Ck.x + Cmir.x), 0.5f * (Ck.y + Cmir.y));
        float2 O = make(0.5f * (Ck.x - Cmir.x), 0.5f * (Ck.y - Cmir.y));
        // Twiddle
        float ang = 2.0f * M_PI * idx / FFTSize;
        float2 W  = make(cosf(ang), sinf(ang));
        float2 W1 = make(-W.y, W.x);
        Zk = cadd(E,cmul(W1,O));
        // Store Zk in register
        res[idx / BlockSize] = Zk;
    }

    __syncthreads();

    // Move results to smem and bit reverse
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        // bit reverse and store in smem
        unsigned int r = isMixedRadix ?
                         mixed_radix_reverse(idx, log4N) :
                         base4_reverse(idx, log4N);
        sdata[r] = res[idx / BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 5a. radix-4 stages inverse FFT
    // ------------------------------------------------
    radix4_CT_DIT<halfN, BlockSize, float2, HandleTypeBwd, false>(sdata, tid, log4N);

    // ------------------------------------------------
    // 5b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix) {

        constexpr int half = halfN >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            float2 a = sdata[k];
            float2 b = sdata[k + half];

            // Twiddle W_N^k
            float2 w = HandleTypeBwd::twiddles()[k];

            float2 t = cmul(w, b);

            sdata[k]        = cadd(a, t);
            sdata[k + half] = csub(a, t);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 6. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx < halfN / 2; idx += BlockSize) {
        float4 val = *(float4 *)(sdata + 2 * idx);
        val.x /= halfN;
        val.y /= halfN;
        val.z /= halfN;
        val.w /= halfN;
        *(float4 *)(odata + 4 * idx) = val;
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleTypeFwd,
typename HandleTypeBwd
>
CUALGO_GLOBAL
void convFFT1dCTKernelMixedRadixDIFDIT(T * CUALGO_RESTRICT input_data1,
                                       T * CUALGO_RESTRICT input_data2,
                                       T * CUALGO_RESTRICT output_data,
                                       int input1_size,
                                       int input2_size,
                                       int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    float2* sdata = SharedMemory<FFTSize * 2 * sizeof(T)>::get();

    float2 *sdata1 = sdata;
    float2 *sdata2 = sdata + halfN;

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    float* idata1 = input_data1 + batch_id * input1_size;
    float* idata2 = input_data2 + batch_id * input2_size;
    float* odata  = output_data + batch_id * FFTSize;
    const int LOG2N = __builtin_ctz(halfN);
    constexpr int log4N = LOG2N >> 1;
    constexpr bool isMixedRadix = (LOG2N & 1) != 0;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store + Padding
    // ------------------------------------------------
    for (int idx = tid; idx < input1_size / 4; idx += BlockSize)
    {
        float4 val4 = *(float4 *)(idata1 + 4 * idx);
        *(float4 *)(sdata1 + 2 * idx) = val4;
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        sdata1[idx] = make(0.0f, 0.0f);
    }

    for (int idx = tid; idx < input2_size / 4; idx += BlockSize)
    {
        float4 val4 = *(float4 *)(idata2 + 4 * idx);
        *(float4 *)(sdata2 + 2 * idx) = val4;
    }

    for (int idx = tid + input1_size / 2; idx < halfN; idx += BlockSize)
    {
        sdata2[idx] = make(0.0f, 0.0f);
    }

    __syncthreads();

    // ------------------------------------------------
    // 2a. radix-4 stages FFT first signal
    // ------------------------------------------------
    for (int len = halfN; len >= 4; len >>= 2)
    {
        int quarter = len >> 2;

        for (int tidx = tid; tidx < halfN >> 2; tidx += BlockSize)
        {
            // Compute indices for this butterfly
            int block = tidx / quarter;
            int k = tidx % quarter;
            int i = block * len + k;

            float2 a0_1 = sdata1[i + 0 * quarter];
            float2 a1_1 = sdata1[i + 1 * quarter];
            float2 a2_1 = sdata1[i + 2 * quarter];
            float2 a3_1 = sdata1[i + 3 * quarter];

            float2 a0_2 = sdata2[i + 0 * quarter];
            float2 a1_2 = sdata2[i + 1 * quarter];
            float2 a2_2 = sdata2[i + 2 * quarter];
            float2 a3_2 = sdata2[i + 3 * quarter];

            // Radix-4 DIF butterfly
            float2 t0_1 = cadd(cadd(a0_1, a2_1), cadd(a1_1, a3_1));       // y0
            float2 t1_1 = cadd(csub(a0_1, a2_1), cmulj(csub(a3_1, a1_1))); // y1
            float2 t2_1 = csub(cadd(a0_1, a2_1), cadd(a1_1, a3_1));       // y2
            float2 t3_1 = csub(csub(a0_1, a2_1), cmulj(csub(a3_1, a1_1))); // y3

            float2 t0_2 = cadd(cadd(a0_2, a2_2), cadd(a1_2, a3_2));       // y0
            float2 t1_2 = cadd(csub(a0_2, a2_2), cmulj(csub(a3_2, a1_2))); // y1
            float2 t2_2 = csub(cadd(a0_2, a2_2), cadd(a1_2, a3_2));       // y2
            float2 t3_2 = csub(csub(a0_2, a2_2), cmulj(csub(a3_2, a1_2))); // y3

            int twiddle_idx = (k * halfN) / len;
            float2 W1 = HandleTypeFwd::twiddles()[twiddle_idx];
            float2 W2 = cmul(W1, W1);
            float2 W3 = cmul(W2, W1);

            sdata1[i + 0 * quarter] = t0_1;
            sdata1[i + 1 * quarter] = cmul(t1_1, W1);
            sdata1[i + 2 * quarter] = cmul(t2_1, W2);
            sdata1[i + 3 * quarter] = cmul(t3_1, W3);

            sdata2[i + 0 * quarter] = t0_2;
            sdata2[i + 1 * quarter] = cmul(t1_2, W1);
            sdata2[i + 2 * quarter] = cmul(t2_2, W2);
            sdata2[i + 3 * quarter] = cmul(t3_2, W3);
        }
        __syncthreads();
    }

    // ------------------------------------------------
    // 2b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix)
    {
        // Mixed radix is still buggy
        constexpr int half = halfN >> 1;

        for (int tidx = tid; tidx < half; tidx += BlockSize)
        {
            float2 w = HandleTypeFwd::twiddles()[0];

            float2 a1 = sdata1[2 * tidx];
            float2 b1 = sdata1[2 * tidx + 1];

            float2 a2 = sdata2[2 * tidx];
            float2 b2 = sdata2[2 * tidx + 1];

            // DIF butterfly (add/sub first)
            float2 t0_1 = cadd(a1, b1);
            float2 t1_1 = csub(a1, b1);

            float2 t0_2 = cadd(a2, b2);
            float2 t1_2 = csub(a2, b2);

            t1_1 = cmul(t1_1, w);
            t1_2 = cmul(t1_2, w);

            sdata1[2 * tidx]     = t0_1;
            sdata1[2 * tidx + 1] = t1_1;

            sdata2[2 * tidx]     = t0_2;
            sdata2[2 * tidx + 1] = t1_2;
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Post-processing FFT + Convolution
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        int r = isMixedRadix ?
                mixed_radix_reverse(idx, log4N) :
                base4_reverse(idx, log4N);
        if(idx < r)
        {
            float2 tmp = sdata1[idx];
            sdata1[idx] = sdata1[r];
            sdata1[r] = tmp;

            tmp = sdata2[idx];
            sdata2[idx] = sdata2[r];
            sdata2[r] = tmp;
        }
    }

    __syncthreads();

    // Post-process FFT and compute convolution (results in registers)
    float2 res[(halfN + 1) / BlockSize + 1];
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        float ang          = -2.0f * M_PI * idx / FFTSize;
        float2 W           = make(cosf(ang), sinf(ang));
        float2 res1, res2;
        // First signal post-process
        {
            float2 Zk          = sdata1[idx % halfN];
            float2 Zk_conj     = conjf2(sdata1[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res1               = cadd(E, cmul(W, O));
        }

        // Second signal post-process
        {
            float2 Zk          = sdata2[idx % halfN];
            float2 Zk_conj     = conjf2(sdata2[(halfN - idx) % halfN]);
            float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
            float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
            res2               = cadd(E, cmul(W, O));
        }

        res[idx / BlockSize] = cmul(res1, res2);
    }

    __syncthreads();

    // Move results to smem
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        sdata[idx] = res[idx/BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 4. Pre-processing iFFT
    // ------------------------------------------------

    // Pre-process inverse FFT and store results in register
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 Zk;
        float2 Ck      = sdata[idx];
        float2 Cmir    = conjf2(sdata[(halfN - idx)]);
        // Compute E and O
        float2 E = make(0.5f * (Ck.x + Cmir.x), 0.5f * (Ck.y + Cmir.y));
        float2 O = make(0.5f * (Ck.x - Cmir.x), 0.5f * (Ck.y - Cmir.y));
        // Twiddle
        float ang = 2.0f * M_PI * idx / FFTSize;
        float2 W  = make(cosf(ang), sinf(ang));
        float2 W1 = make(-W.y, W.x);
        Zk = cadd(E,cmul(W1,O));
        // Store Zk in register
        res[idx / BlockSize] = Zk;
    }

    __syncthreads();

    // Move results to smem and bit reverse
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        // bit reverse and store in smem
        unsigned int r = isMixedRadix ?
                         mixed_radix_reverse(idx, log4N) :
                         base4_reverse(idx, log4N);
        sdata[r] = res[idx / BlockSize];
    }

    __syncthreads();

    // ------------------------------------------------
    // 5a. radix-4 stages inverse FFT
    // ------------------------------------------------
    radix4_CT_DIT<halfN, BlockSize, float2, HandleTypeBwd, false>(sdata, tid, log4N);

    // ------------------------------------------------
    // 5b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix) {

        constexpr int half = halfN >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            float2 a = sdata[k];
            float2 b = sdata[k + half];

            // Twiddle W_N^k
            float2 w = HandleTypeBwd::twiddles()[k];

            float2 t = cmul(w, b);

            sdata[k]        = cadd(a, t);
            sdata[k + half] = csub(a, t);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 6. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx < halfN / 2; idx += BlockSize) {
        float4 val = *(float4 *)(sdata + 2 * idx);
        val.x /= halfN;
        val.y /= halfN;
        val.z /= halfN;
        val.w /= halfN;
        *(float4 *)(odata + 4 * idx) = val;
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform 1d C2R ifft
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
    void convFFT1dCT(T *idata1,
                     T *idata2,
                     T *odata,
                     int input1_size,
                     int input2_size,
                     int batch_size,
                     cudaStream_t stream = 0,
                     bool async = false)
    {
        static_assert(is_power_of_two<FFTSize>());

        dim3 blocksPerGrid3(batch_size, 1, 1);
        dim3 threadsPerBlock3(BlockSize, 1, 1);
        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        constexpr bool useConst = fitInConstantMemory<2 * FFTSize, T>();
        using fftHandle = fftHandle<FFTSize / 2, useConst>;
        using ifftHandle = ifftHandle<FFTSize / 2, useConst>;

        constexpr int SmemBytes = FFTSize * 2 * sizeof(T);
        if constexpr(SmemBytes > STATIC_SMEM_LIMIT)
        {
            check_cuda(cudaFuncSetAttribute(
                convFFT1dCTKernelMixedRadixDITDIT<FFTSize, BlockSize, T,
                    fftHandle, ifftHandle>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SmemBytes
            ));

            TIME(blocksPerGrid3, threadsPerBlock3, SmemBytes, stream, async, 
            CUALGO_KERNEL_NAME(
                convFFT1dCTKernelMixedRadixDITDIT<FFTSize, BlockSize, T,
                fftHandle, ifftHandle>),
            idata1, idata2, odata, input1_size, input2_size, batch_size);
        }
        else
        {
            TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                CUALGO_KERNEL_NAME(
                    convFFT1dCTKernelMixedRadixDITDIT<FFTSize, BlockSize, T,
                    fftHandle, ifftHandle>),
                idata1, idata2, odata, input1_size, input2_size, batch_size);
        }
    }
}

#endif
