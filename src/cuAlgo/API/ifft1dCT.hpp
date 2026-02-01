/*
 * @file ifft1dCT.hpp
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

#ifndef IFFT1DCT_HPP
#define IFFT1DCT_HPP

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
void ifft1dCTKernelRadix2DIT(T * CUALGO_RESTRICT input_data,
                             T * CUALGO_RESTRICT output_data,
                             int batch_size)
{
    CUALGO_SHMEM T sdata[FFTSize];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;
    const int LOGN = __builtin_ctz(FFTSize);

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store
    // ------------------------------------------------
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        unsigned int r = base2_reverse(idx, LOGN);
        sdata[r] = idata[idx];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    radix2_CT_DIT<FFTSize, BlockSize, T, HandleType>(sdata, tid, LOGN);

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    float invN = 1.0f / FFTSize;
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        odata[idx].x = sdata[idx].x * invN;
        odata[idx].y = sdata[idx].y * invN;
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform 1d C2C ifft
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
    void ifft1dCT(T *idata,
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
                ifft1dCTKernelRadix2DIT<FFTSize, BlockSize, T, ifftHandle<FFTSize>>),
            idata, odata, batch_size);
    }
}

#endif
