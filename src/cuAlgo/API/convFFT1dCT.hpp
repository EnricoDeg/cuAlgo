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

    CUALGO_SHMEM float2 sdata[2 * halfN];

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

        TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
            CUALGO_KERNEL_NAME(
                convFFT1dCTKernelRadix2DITDIT<FFTSize, BlockSize, T,
                fftHandle<FFTSize / 2>, ifftHandle<FFTSize / 2>>),
            idata1, idata2, odata, input1_size, input2_size, batch_size);
    }
}

#endif
