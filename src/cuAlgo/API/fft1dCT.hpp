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

#ifndef FFT1DCT_HPP
#define FFT1DCT_HPP

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
void fft1dCTKernelRadix8DIT(T * CUALGO_RESTRICT input_data,
                            T * CUALGO_RESTRICT output_data,
                            int batch_size)
{
    CUALGO_SHMEM T smem[FFTSize];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    T* idata = input_data  + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    constexpr int log8N = __builtin_ctz(FFTSize) / 3;

    // --------------------------------------------------
    // 1) Load global -> shared + Base-8 digit reversal
    // --------------------------------------------------
    for (int i = tid; i < FFTSize; i += BlockSize) {
        unsigned int r = base8_reverse(i, log8N);
        smem[r] = idata[i];
    }
    __syncthreads();

    // --------------------------------------------------
    // 2) Radix-8 FFT stages
    // --------------------------------------------------
    radix8_CT_DIT<FFTSize, BlockSize, T, HandleType>(smem, tid, log8N);

    // --------------------------------------------------
    // 3) Store shared -> global
    // --------------------------------------------------
    for (int i = tid; i < FFTSize; i += BlockSize) {
        odata[i] = smem[i];
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void fft1dCTKernelRadix2DIT(T * CUALGO_RESTRICT input_data,
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
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        odata[idx] = sdata[idx];
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType>
CUALGO_GLOBAL
void fft1dCTKernelMixedRadixDIT(T * CUALGO_RESTRICT input_data,
                                T * CUALGO_RESTRICT output_data,
                                int batch_size)
{
    CUALGO_SHMEM T smem[FFTSize];

    constexpr int LOG2N = __builtin_ctz(FFTSize);
    constexpr int log4N = LOG2N >> 1;
    constexpr bool isMixedRadix = (LOG2N & 1) != 0;

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    // Pointer shift to correct batch
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    // ------------------------------------------------
    // 1. Load + Base-4 digit-reversed store
    // ------------------------------------------------
    for (int base = tid; base < FFTSize; base += BlockSize) {
        unsigned r = isMixedRadix ?
                     mixed_radix_reverse(base, log4N) :
                     base4_reverse(base, log4N);
        smem[r] = idata[base];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2a. radix-4 stages
    // ------------------------------------------------
    radix4_CT_DIT<FFTSize, BlockSize, T, HandleType, true>(smem, tid, log4N);

    // ------------------------------------------------
    // 2b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix) {

        constexpr int half = FFTSize >> 1;

        for (int k = tid; k < half; k += BlockSize) {

            T a = smem[k];
            T b = smem[k + half];

            // Twiddle W_N^k
            T w = HandleType::twiddles()[k];

            T t = cmul(w, b);

            smem[k]        = cadd(a, t);
            smem[k + half] = csub(a, t);
        }

        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    for (int base = tid; base < FFTSize; base += BlockSize) {
        odata[base] = smem[base];
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
unsigned int Tile,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void fft1dCTKernelRadix2Stage1DIT(T * CUALGO_RESTRICT input_data,
                         T * CUALGO_RESTRICT output_data,
                         int batch_size)
{
    CUALGO_SHMEM T sdata[Tile];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data + batch_id * Tile;
    T* odata = output_data + batch_id * Tile;
    const int LOGN = __builtin_ctz(FFTSize);

    // ------------------------------------------------
    // 1. Load (Assume data is already bit reversed)
    // ------------------------------------------------
    for (int idx = tid; idx < Tile; idx += BlockSize)
    {
        sdata[idx] = idata[idx];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    radix4_CT_DIT<FFTSize, BlockSize, T, HandleType>(sdata, tid, LOGN);

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    for (int idx = tid; idx < Tile; idx += BlockSize)
    {
        odata[idx] = sdata[idx];
    }
}

template<
unsigned int N,
unsigned int BlockSize,
typename T,
typename HandleType>
CUALGO_GLOBAL
void fft1dCTMergeKernelDIT(T* data, int stage)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;
    int half = m >> 1;
    int total = N >> 1;
    if(tid >= total) return;

    int k = tid / half;
    int j = tid % half;

    int i1 = k*m + j;
    int i2 = i1 + half;

    float angle = -2.0f * M_PI * j / m;
    T w = make(cosf(angle), sinf(angle));

    T u = data[i1];
    T t = cmul(w, data[i2]);

    data[i1] = cadd(u,t);
    data[i2] = csub(u,t);
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void fft1dCTKernelRadix2DIF(T * CUALGO_RESTRICT input_data,
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
    // 1. Load (No bit reversal needed)
    // ------------------------------------------------
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        sdata[idx] = idata[idx];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    radix2_CT_DIF<FFTSize, BlockSize, T, HandleType>(sdata, tid);

    // ------------------------------------------------
    // 3. Store + bit reversal
    // ------------------------------------------------
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        int r = base2_reverse(idx, LOGN);
        odata[r] = sdata[idx];
    }
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void fft1dCTKernelMixedRadixDIF(T * CUALGO_RESTRICT input_data,
                                T * CUALGO_RESTRICT output_data,
                                int batch_size)
{
    CUALGO_SHMEM T sdata[FFTSize];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;
    constexpr int LOG2N = __builtin_ctz(FFTSize);
    constexpr bool isMixedRadix = (LOG2N & 1) != 0;
    constexpr int log4N = LOG2N >> 1;

    // ------------------------------------------------
    // 1. Load (No bit reversal needed)
    // ------------------------------------------------
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        sdata[idx] = idata[idx];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-4 stages
    // ------------------------------------------------
    radix4_CT_DIF<FFTSize, BlockSize, T, HandleType, true>(sdata, tid);

    // ------------------------------------------------
    // 2b. final radix-2 stage (only if FFTSize has odd log2)
    // ------------------------------------------------
    if constexpr(isMixedRadix)
    {
        for (int tidx = tid; tidx < FFTSize / 2; tidx += BlockSize)
        {
            T w = HandleType::twiddles()[0];

            T a = sdata[2 * tidx];
            T b = sdata[2 * tidx + 1];

            // DIF butterfly (add/sub first)
            T t0 = cadd(a, b);
            T t1 = csub(a, b);

            t1 = cmul(t1, w);

            sdata[2 * tidx]     = t0;
            sdata[2 * tidx + 1] = t1;
        }
        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Store + bit reversal
    // ------------------------------------------------
    for (int idx = tid; idx < FFTSize; idx += BlockSize)
    {
        int r = isMixedRadix ?
                mixed_radix_reverse(idx, log4N) :
                base4_reverse(idx, log4N);
        odata[r] = sdata[idx];
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
    void fft1dCT(T *idata,
                 T *odata,
                 int batch_size,
                 cudaStream_t stream = 0,
                 bool async = false)
    {
        static_assert(is_power_of_two<FFTSize>());

        if constexpr(FFTSize > 4096)
        {
            constexpr unsigned int Tile = 4096;
            constexpr int blocks_per_fft = FFTSize / Tile;

            int blocks = (FFTSize + BlockSize - 1) / BlockSize;

            TIME(blocks, BlockSize, 0, stream, async, 
                CUALGO_KERNEL_NAME(
                    bit_reverse_global<FFTSize, T>),
                idata);

            dim3 blocksPerGrid3(batch_size * blocks_per_fft, 1, 1);
            dim3 threadsPerBlock3(BlockSize, 1, 1);
            print_kernel_config(threadsPerBlock3, blocksPerGrid3);

            TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                CUALGO_KERNEL_NAME(
                    fft1dCTKernelRadix2Stage1DIT<FFTSize, BlockSize, Tile, T, fftHandle<FFTSize>>),
                idata, odata, batch_size);

            int total_butterflies = FFTSize / 2;
            blocks = (total_butterflies + BlockSize - 1) / BlockSize;
            int log2N = (int)log2f((float)FFTSize);
            int log2Tile = (int)log2f((float)Tile);

            for(int stage = log2Tile+1; stage <= log2N; stage++){
                TIME(blocks, BlockSize, 0, stream, async, 
                    CUALGO_KERNEL_NAME(
                        fft1dCTMergeKernelDIT<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
                    odata, stage);
            }
        }
        else
        {
            dim3 blocksPerGrid3(batch_size, 1, 1);
            dim3 threadsPerBlock3(BlockSize, 1, 1);
            print_kernel_config(threadsPerBlock3, blocksPerGrid3);

            if constexpr(is_power_of_eight<FFTSize>() && FFTSize <= 512)
            {
                TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                    CUALGO_KERNEL_NAME(
                        fft1dCTKernelRadix8DIT<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
                    idata, odata, batch_size);
            }
            else
            {
                TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                    CUALGO_KERNEL_NAME(
                        fft1dCTKernelMixedRadixDIT<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
                    idata, odata, batch_size);
            }
        }
    }
}

#endif
