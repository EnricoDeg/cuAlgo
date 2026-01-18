/*
 * @file fft1dStockham.hpp
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

#ifndef FFT1DSTOCKHAM_HPP
#define FFT1DSTOCKHAM_HPP

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/fft.hpp"

template <unsigned int N, typename T>
CUALGO_GLOBAL
void fft1dStockhamKernel(T * CUALGO_RESTRICT idata,
                         T * CUALGO_RESTRICT odata)
{
    extern CUALGO_SHMEM T smem[];

    T* buf0 = smem;
    T* buf1 = smem + N;

    int tid = threadIdx.x;

    // Load input to shared memory
    if (tid < N) {
        buf0[tid] = idata[tid];
    }
    __syncthreads();

    T* in  = buf0;
    T* out = buf1;

    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

    // -------- Stockham stages --------
    for (int s = 0; s < log2N; ++s) {
        int m  = 1 << (s + 1);
        int mh = m >> 1;

        // Each thread computes exactly one output element
        if (tid < N) {
            int group = tid / m;
            int j     = tid % mh;

            // ---- INPUT PERMUTATION (autosort) ----
            int i0 = group * mh + j;
            int i1 = i0 + N / 2;

            T a = in[i0];
            T b = in[i1];

            T w = twiddle(j, m);
            T t = cmul(b, w);

            // ---- LINEAR OUTPUT WRITE ----
            if ((tid % m) < mh)
                out[tid] = cadd(a, t);
            else
                out[tid] = csub(a, t);
        }

        __syncthreads();

        // Ping-pong buffers
        T* tmp = in;
        in = out;
        out = tmp;
    }

    // Write back to global memory
    if (tid < N) {
        odata[tid] = in[tid];
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
    typename T>
    void fft1dStockham(T *idata ,
                       T *odata ,
                       cudaStream_t stream = 0,
                       bool async = false)
    {
        dim3 blocksPerGrid3(1, 1, 1);
        dim3 threadsPerBlock3(FFTSize, 1, 1);

        print_kernel_config(threadsPerBlock3, blocksPerGrid3);

        int shmem_size = 2 * FFTSize * sizeof(T);

        TIME(blocksPerGrid3, threadsPerBlock3, shmem_size, stream, async, 
             CUALGO_KERNEL_NAME(fft1dStockhamKernel<FFTSize>),
             idata, odata);
    }
}

#endif
