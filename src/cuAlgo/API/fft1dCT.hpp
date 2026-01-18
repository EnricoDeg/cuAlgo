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

template <unsigned int FFTSize, typename T, typename HandleType>
CUALGO_GLOBAL
void fft1dCTKernelRadix2(T * CUALGO_RESTRICT idata,
                         T * CUALGO_RESTRICT odata)
{
    extern CUALGO_SHMEM T sdata[];

    int tid = threadIdx.x;
    const int LOGN = __ffs(FFTSize) - 1;

    // ------------------------------------------------
    // 1. Load + Base-2 digit-reversed store
    // ------------------------------------------------
    if (tid < FFTSize / 2)
    {
        unsigned int r;
        r = base2_reverse(2 * tid, LOGN);
        sdata[r] = idata[2 * tid];
        r = base2_reverse(2 * tid + 1, LOGN);
        sdata[r] = idata[2 * tid + 1];
    }
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-2 stages
    // ------------------------------------------------
    for (int len = 2; len <= FFTSize; len <<= 1) {
        int half = len >> 1;
        int block = tid >> (__ffs(len) - 2); // tid / half;
        int k = tid & (half - 1); //tid % half;
        int i = block * len + k;

        if (i + half < FFTSize) {
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
    if (tid < FFTSize)
    {
        odata[2 * tid]     = sdata[2 * tid];
        odata[2 * tid + 1] = sdata[2 * tid + 1];
    }
}

template <unsigned int FFTSize, typename T, typename HandleType>
CUALGO_GLOBAL
void fft1dCTKernelRadix4(T * CUALGO_RESTRICT idata,
                         T * CUALGO_RESTRICT odata)
{
    extern CUALGO_SHMEM T smem[];
    int tid = threadIdx.x;
    constexpr int LOG2N = __builtin_ctz(FFTSize);
    constexpr int log4N = LOG2N >> 1;

    // ------------------------------------------------
    // 1. Load + Base-4 digit-reversed store
    // ------------------------------------------------
    unsigned r = base4_reverse(4 * tid, log4N);
    smem[r] = idata[4 * tid];
    r = base4_reverse(4 * tid + 1, log4N);
    smem[r] = idata[4 * tid + 1];
    r = base4_reverse(4 * tid + 2, log4N);
    smem[r] = idata[4 * tid + 2];
    r = base4_reverse(4 * tid + 3, log4N);
    smem[r] = idata[4 * tid + 3];
    __syncthreads();

    // ------------------------------------------------
    // 2. radix-4 stages
    // ------------------------------------------------
    for (int stage = 0, m = 4; stage < log4N; ++stage, m <<= 2) {

        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;

        int p = k * m + j;

        float angle = -2.0f * M_PI * j / m;
        float s, c;
        __sincosf(angle, &s, &c);
        T W1 = make(c, s);
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

        __syncthreads();
    }

    // ------------------------------------------------
    // 3. Store (natural order)
    // ------------------------------------------------
    odata[4 * tid] = smem[4 * tid];
    odata[4 * tid + 1] = smem[4 * tid + 1];
    odata[4 * tid + 2] = smem[4 * tid + 2];
    odata[4 * tid + 3] = smem[4 * tid + 3];
}

namespace cuAlgo {

    template <unsigned int N>
    __device__ __constant__ float2 fft_twiddles[N / 2];

    template<unsigned int N>
    struct fftHandle
    {
        __device__ static const float2* twiddles() {
            return fft_twiddles<N>;
        }
    };

    void create_twiddles(float2* h_twiddles, int N)
    {
        for (int k = 0; k < N / 2; k++) {
            float angle = -2.0f * M_PI * k / N;
            h_twiddles[k].x = cosf(angle);
            h_twiddles[k].y = sinf(angle);
        }
    }

    template<unsigned int BlockSize>
    void fft1dCT_plan()
    {
        float2* h_twiddles = new float2[BlockSize/2];
        create_twiddles(h_twiddles, BlockSize);
        cudaMemcpyToSymbol(fft_twiddles<BlockSize>,
                           h_twiddles,
                           (BlockSize/2) * sizeof(float2));
    }

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
    typename T>
    void fft1dCT(T *idata ,
                 T *odata ,
                 cudaStream_t stream = 0,
                 bool async = false)
    {
        dim3 blocksPerGrid3(1, 1, 1);
        dim3 threadsPerBlock3(FFTSize, 1, 1);

        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        if constexpr(is_power_of_four<FFTSize>())
        {
            std::cout << "Running Radix4\n";
            constexpr unsigned int RadixValue = 4;

            dim3 blocksPerGrid3(1, 1, 1);
            dim3 threadsPerBlock3(FFTSize / RadixValue, 1, 1);

            print_kernel_config(threadsPerBlock3, blocksPerGrid3);

            int shmem_size = FFTSize * sizeof(T);

            fft1dCT_plan<FFTSize>();

            TIME(blocksPerGrid3, threadsPerBlock3, shmem_size, stream, async, 
                 CUALGO_KERNEL_NAME(fft1dCTKernelRadix4<FFTSize, T, fftHandle<FFTSize>>),
                 idata, odata);
        }
        else if constexpr(is_power_of_two<FFTSize>())
        {
            std::cout << "Running Radix2\n";
            constexpr unsigned int RadixValue = 2;

            dim3 blocksPerGrid3(1, 1, 1);
            dim3 threadsPerBlock3(FFTSize / RadixValue, 1, 1);

            print_kernel_config(threadsPerBlock3, blocksPerGrid3);

            int shmem_size = FFTSize * sizeof(T);

            fft1dCT_plan<FFTSize>();

            TIME(blocksPerGrid3, threadsPerBlock3, shmem_size, stream, async, 
                 CUALGO_KERNEL_NAME(fft1dCTKernelRadix2<FFTSize, T, fftHandle<FFTSize>>),
                 idata, odata);
        }
    }
}

#endif
