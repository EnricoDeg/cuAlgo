/*
 * @file fft1dCTr2c.hpp
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

#ifndef FFT1DCTR2C_HPP
#define FFT1DCTR2C_HPP

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
void fft1dCTr2cKernelRadix2(T * CUALGO_RESTRICT input_data,
                            T * CUALGO_RESTRICT output_data,
                            int batch_size)
{
    constexpr int halfN = FFTSize / 2;

    CUALGO_SHMEM float2 sdata[halfN];

    int tid = threadIdx.x;
    T* idata = input_data;
    T* odata = output_data;
    const int LOGN = __ffs(halfN) - 1;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store
    // ------------------------------------------------
    for (int idx = tid; idx < halfN; idx += BlockSize)
    {
        float2 val = make(idata[2 * idx], idata[2 * idx + 1]);
        unsigned int r = base2_reverse(idx, LOGN);
        sdata[r] = val;
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    radix2_CT<halfN, BlockSize, float2, HandleType>(sdata, tid, LOGN);

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx <= halfN; idx += BlockSize)
    {
        float2 Zk          = sdata[idx % halfN];
        float2 Zk_conj     = conjf2(sdata[(halfN - idx) % halfN]);
        float2 E           = make(0.5f * (Zk.x + Zk_conj.x), 0.5f * (Zk.y + Zk_conj.y));
        float2 O           = make(0.5f * (Zk.y - Zk_conj.y), 0.5f * (Zk_conj.x - Zk.x));
        float ang          = -2.0f * M_PI * idx / FFTSize;
        float2 W           = make(cosf(ang), sinf(ang));
        float2 res         = cadd(E, cmul(W, O));
        odata[2 * idx]     = res.x;
        odata[2 * idx + 1] = res.y;
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform 1d R2C fft
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
    void fft1dCTr2c(T *idata,
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
                fft1dCTr2cKernelRadix2<FFTSize, BlockSize, T, fftHandle<FFTSize / 2>>),
            idata, odata, batch_size);
    }
}

#endif
