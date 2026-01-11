/*
 * @file fft1dCT.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
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
#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"


CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

CUALGO_HOST_DEVICE CUALGO_FORCE_INLINE float2 cmul(float2 a, float2 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

template <unsigned int FFTSize, typename T, typename HandleType>
CUALGO_GLOBAL
void fft1dCTKernel(T * CUALGO_RESTRICT idata,
                   T * CUALGO_RESTRICT odata)
{
    extern CUALGO_SHMEM T sdata[];

    int tid = threadIdx.x;

    // Load to shared memory
    if (tid < FFTSize)
        sdata[tid] = idata[tid];
    __syncthreads();

    // Bit-reversal permutation
    unsigned int x = tid;
    unsigned int j = 0;
    const int LOGN = __ffs(FFTSize) - 1;

    for (int i = 0; i < LOGN; ++i) {
        j = (j << 1) | (x & 1);
        x >>= 1;
    }

    if (j > tid) {
        T tmp = sdata[tid];
        sdata[tid] = sdata[j];
        sdata[j] = tmp;
    }
    __syncthreads();

    // FFT stages
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

    // Store back
    if (tid < FFTSize)
        odata[tid] = sdata[tid];
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
    unsigned int BlockSize,
    typename T>
    void fft1dCT(T *idata ,
                 T *odata ,
                 cudaStream_t stream = 0,
                 bool async = false) {

        dim3 blocksPerGrid3(1, 1, 1);
        dim3 threadsPerBlock3(BlockSize, 1, 1);

        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        int shmem_size = BlockSize * sizeof(T);

        fft1dCT_plan<BlockSize>();

        TIME(blocksPerGrid3, threadsPerBlock3, shmem_size, stream, async, 
             CUALGO_KERNEL_NAME(fft1dCTKernel<BlockSize, T, fftHandle<BlockSize>>),
             idata, odata);
    }
}
