/*
 * @file fft1dPlan.hpp
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

#ifndef FFT1DPLAN_HPP
#define FFT1DPLAN_HPP

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

    template<unsigned int FFTSize>
    void fft1dCT_plan()
    {
        float2* h_twiddles = new float2[FFTSize/2];
        create_twiddles(h_twiddles, FFTSize);
        cudaMemcpyToSymbol(fft_twiddles<FFTSize>,
                           h_twiddles,
                           (FFTSize/2) * sizeof(float2));
    }
}

#endif
