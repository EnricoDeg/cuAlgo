/*
 * @file ifft1dCTc2r.hpp
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

#ifndef IFFT1DCTC2R_HPP
#define IFFT1DCTC2R_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"
#include "cuAlgo/API/fft1dPlan.hpp"

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void ifft1dCTc2rKernelRadix2DIT(T * CUALGO_RESTRICT input_data,
                                T * CUALGO_RESTRICT output_data,
                                int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    CUALGO_SHMEM float2 sdata[halfN];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data  + batch_id * (FFTSize + 2);
    T* odata = output_data + batch_id * (FFTSize);
    const int LOGN = __builtin_ctz(halfN);

    // ------------------------------------------------
    // 2. Preprocess input and prefer for CT stages
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 Zk;
        float2 Ck      = make(idata[2 * idx], idata[2 * idx + 1]);
        float2 Cmir    = conjf2(make(idata[2 * (halfN - idx)],
                                     idata[2 * (halfN - idx) + 1])); // mirror element
        // Compute E and O
        float2 E = make(0.5f * (Ck.x + Cmir.x), 0.5f * (Ck.y + Cmir.y));
        float2 O = make(0.5f * (Ck.x - Cmir.x), 0.5f * (Ck.y - Cmir.y));
        // Twiddle
        float ang = 2.0f * M_PI * idx / FFTSize;
        float2 W  = make(cosf(ang), sinf(ang));
        float2 W1 = make(-W.y, W.x);
        Zk = cadd(E,cmul(W1,O));
        // Store Zk in length-N/2 array
        unsigned int r = base2_reverse(idx, LOGN);
        sdata[r] = Zk;
    }

    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    radix2_CT_DIT<halfN, BlockSize, float2, HandleType>(sdata, tid, LOGN);

    // ------------------------------------------------
    // 3. Store (natural order)
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
typename HandleType
>
CUALGO_GLOBAL
void ifft1dCTc2rKernelMixedRadixDIT(T * CUALGO_RESTRICT input_data,
                                    T * CUALGO_RESTRICT output_data,
                                    int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    CUALGO_SHMEM float2 sdata[halfN];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data  + batch_id * (FFTSize + 2);
    T* odata = output_data + batch_id * (FFTSize);
    constexpr int LOG2N = __builtin_ctz(halfN);
    constexpr int log4N = LOG2N >> 1;
    constexpr bool isMixedRadix = (LOG2N & 1) != 0;

    // ------------------------------------------------
    // 2. Preprocess input and prefer for CT stages
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 Zk;
        float2 Ck      = make(idata[2 * idx], idata[2 * idx + 1]);
        float2 Cmir    = conjf2(make(idata[2 * (halfN - idx)], idata[2 * (halfN - idx) + 1]));      // mirror element
        // Compute E and O
        float2 E = make(0.5f * (Ck.x + Cmir.x), 0.5f * (Ck.y + Cmir.y));
        float2 O = make(0.5f * (Ck.x - Cmir.x), 0.5f * (Ck.y - Cmir.y));
        // Twiddle
        float ang = 2.0f * M_PI * idx / FFTSize;
        float2 W  = make(cosf(ang), sinf(ang));
        float2 W1 = make(-W.y, W.x);
        Zk = cadd(E,cmul(W1,O));
        // Store Zk in length-N/2 array
        unsigned int r = isMixedRadix ?
                         mixed_radix_reverse(idx, log4N) :
                         base4_reverse(idx, log4N);
        sdata[r] = Zk;
    }

    __syncthreads();

    // ------------------------------------------------
    // 2. radix-4 stages
    // ------------------------------------------------
    radix4_CT_DIT<halfN, BlockSize, float2, HandleType, false>(sdata, tid, log4N);

    // ------------------------------------------------
    // 2b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix) {

        constexpr int half = halfN >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            float2 a = sdata[k];
            float2 b = sdata[k + half];

            // Twiddle W_N^k
            float2 w = HandleType::twiddles()[k];

            float2 t = cmul(w, b);

            sdata[k]        = cadd(a, t);
            sdata[k + half] = csub(a, t);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize) {
        odata[2*idx]     = sdata[idx].x / (FFTSize / 2);  // even samples, scale
        odata[2*idx + 1] = sdata[idx].y / (FFTSize / 2);  // odd samples
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
    void ifft1dCTc2r(T *idata,
                     T *odata,
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
                ifft1dCTc2rKernelMixedRadixDIT<FFTSize, BlockSize, T, ifftHandle<FFTSize / 2>>),
            idata, odata, batch_size);
    }
}

#endif
