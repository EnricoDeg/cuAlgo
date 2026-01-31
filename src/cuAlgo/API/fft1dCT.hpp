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

__device__ __forceinline__ int log8N(int N)
{
    int log8 = 0;
    while (N > 1) {
        N >>= 3;   // divide by 8
        log8++;
    }
    return log8;
}

template<
unsigned int FFTSize,
unsigned int BlockSize,
typename T,
typename HandleType
>
CUALGO_GLOBAL
void fft1dCTKernelRadix8(T * CUALGO_RESTRICT input_data,
                         T * CUALGO_RESTRICT output_data,
                         int batch_size)
{
    CUALGO_SHMEM T smem[FFTSize];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;

    T* idata = input_data  + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;

    const int log8N_val = log8N(FFTSize);

    // --------------------------------------------------
    // 1) Load global -> shared + Base-8 digit reversal
    // --------------------------------------------------
    for (int i = tid; i < FFTSize; i += BlockSize) {
        unsigned int r = base8_reverse(i, log8N_val);
        smem[r] = idata[i];
    }
    __syncthreads();

    // --------------------------------------------------
    // 2) Radix-8 FFT stages
    // --------------------------------------------------
    for (int span = 8; span <= FFTSize; span *= 8) {

        int stride  = span >> 3;
        int butterflies = (FFTSize / span) * stride;
        int tw_step = FFTSize / span;
        for (int base = tid; base < FFTSize >> 3; base += BlockSize)
        {
            if (base < butterflies) {
                int group = base / stride;
                int k     = base % stride;
                int base  = group * span + k;
                int tw    = k * tw_step;

                radix8_butterfly<HandleType>(&smem[base], stride, tw, FFTSize);
            }
        }
        __syncthreads();
    }

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
void fft1dCTKernelRadix2(T * CUALGO_RESTRICT input_data,
                         T * CUALGO_RESTRICT output_data,
                         int batch_size)
{
    CUALGO_SHMEM T sdata[FFTSize];

    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    T* idata = input_data + batch_id * FFTSize;
    T* odata = output_data + batch_id * FFTSize;
    const int LOGN = __ffs(FFTSize) - 1;

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
    for (int stage = 0, len = 2; stage < LOGN; ++stage, len <<= 1) {

        int half = len >> 1;

        for (int tidx = tid; tidx < FFTSize / 2; tidx += BlockSize)
        {
            int block = tidx / half; //>> (__ffs(len) - 2); // tidx / half;
            int k = tidx % half; //& (half - 1); //tidx % half;
            int i = block * len + k;

            int twiddle_idx = (k * FFTSize) / len;
            T w = HandleType::twiddles()[twiddle_idx];
            T u = sdata[i];
            T v = cmul(w, sdata[i + half]);

            sdata[i]       = cadd(u, v);
            sdata[i + half]= csub(u, v);
        }
        __syncthreads();
    }

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
void fft1dCTKernelMixedRadix(T * CUALGO_RESTRICT input_data,
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
    for (int stage = 0, m = 4; stage < log4N; ++stage, m <<= 2)
    {

        int quarter = m >> 2;

        for (int base = tid; base < FFTSize >> 2; base += BlockSize)
        {
            int j = base % quarter;
            int k = base / quarter;
            int p = k * m + j;

            int twiddle_idx = (j * FFTSize) / m;
            T W1 = HandleType::twiddles()[twiddle_idx];
            T W2 = cmul(W1, W1);
            T W3 = cmul(W2, W1);

            T x0 = smem[p + 0 * quarter];
            T x1 = cmul(W1, smem[p + 1 * quarter]);
            T x2 = cmul(W2, smem[p + 2 * quarter]);
            T x3 = cmul(W3, smem[p + 3 * quarter]);

            T t0 = cadd(x0, x2);
            T t1 = cadd(x1, x3);
            T t2 = csub(x0, x2);
            T t3 = csub(x1, x3);

            smem[p + 0 * quarter] = cadd(t0, t1);
            smem[p + 2 * quarter] = csub(t0, t1);

            smem[p + 1 * quarter] =
                make(t2.x + t3.y, t2.y - t3.x);
            smem[p + 3 * quarter] =
                make(t2.x - t3.y, t2.y + t3.x);
        }

        __syncthreads();
    }

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

template<unsigned int FFTSize, typename T>
__global__ void bit_reverse_global(T* data)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FFTSize) return;

    unsigned int r = base2_reverse(i, __ffs(FFTSize) - 1);

    if (r > i) {
        T tmp = data[i];
        data[i] = data[r];
        data[r] = tmp;
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
    const int LOGN = __ffs(Tile) - 1;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store
    // ------------------------------------------------
    for (int idx = tid; idx < Tile; idx += BlockSize)
    {
        // unsigned int r = base2_reverse(idx, LOGN);
        sdata[idx] = idata[idx];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    for (int stage = 0, len = 2; stage < LOGN; ++stage, len <<= 1) {

        int half = len >> 1;

        for (int tidx = tid; tidx < Tile / 2; tidx += BlockSize)
        {
            int block = tidx / half; //>> (__ffs(len) - 2); // tidx / half;
            int k = tidx % half; //& (half - 1); //tidx % half;
            int i = block * len + k;

            int twiddle_idx = (k * FFTSize) / len;
            T w = HandleType::twiddles()[twiddle_idx];

            T u = sdata[i];
            T v = cmul(w, sdata[i+half]);

            sdata[i]       = cadd(u,v);
            sdata[i+half]  = csub(u,v);
        }
        __syncthreads();
    }

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
void fft1dCTMergeKernelDIT(T* data, int stage){
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
    float2 w = make(cosf(angle), sinf(angle));

    float2 u = data[i1];
    float2 t = cmul(w, data[i2]);

    data[i1] = cadd(u,t);
    data[i2] = csub(u,t);
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
                        fft1dCTKernelRadix8<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
                    idata, odata, batch_size);
            }
            else
            {
                TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                    CUALGO_KERNEL_NAME(
                        fft1dCTKernelMixedRadix<FFTSize, BlockSize, T, fftHandle<FFTSize>>),
                    idata, odata, batch_size);
            }
        }
    }
}

#endif
