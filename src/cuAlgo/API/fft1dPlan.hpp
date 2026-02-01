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

    constexpr int CONST_MEM_LIMIT = 64 * 1024;

    template<unsigned int Size, typename T>
    constexpr bool fitInConstantMemory()
    {
        return (Size * sizeof(T)) < CONST_MEM_LIMIT;
    }

    template <unsigned int N>
    __device__ __constant__ float2 fft_twiddles[N];

    template <unsigned int N>
    __device__ __constant__ float2 ifft_twiddles[N];

    template<unsigned int N, bool IsFwd, bool UseConst>
    struct TwiddleAccessor;

    template<unsigned int N, bool IsFwd>
    struct TwiddleAccessor<N, IsFwd, true>
    {
        __device__ float2 operator[](int i) const
        {
            if constexpr(IsFwd)
                return fft_twiddles<N>[i];
            else
                return ifft_twiddles<N>[i];
        }
    };

    template<int Factor>
    __device__ inline float2 compute_twiddle(int k, int N)
    {
        float angle = Factor * 2.f * M_PI * k / N;
        return make_float2(cosf(angle), sinf(angle));
    }

    template<unsigned int N, bool IsFwd>
    struct TwiddleAccessor<N, IsFwd, false>
    {
        static constexpr int Factor = IsFwd ? -1 : 1;
        __device__ float2 operator[](int i) const
        {
            return compute_twiddle<Factor>(i, N);
        }
    };

    template<unsigned int N, bool UseConst = true>
    struct fftHandle
    {
        using Accessor = TwiddleAccessor<N, true, UseConst>;

        __device__ static Accessor twiddles()
        {
            return Accessor{};
        }
    };

    template<unsigned int N, bool UseConst = true>
    struct ifftHandle
    {
        using Accessor = TwiddleAccessor<N, false, UseConst>;

        __device__ static Accessor twiddles()
        {
            return Accessor{};
        }
    };

    template<int Factor>
    void create_twiddles(float2* h_twiddles, int N)
    {
        for (int k = 0; k < N; k++) {
            float angle = Factor * 2.0f * M_PI * k / N;
            h_twiddles[k].x = cosf(angle);
            h_twiddles[k].y = sinf(angle);
        }
    }

    template<unsigned int FFTSize, bool AvoidConst = false>
    void fft1dCT_plan()
    {
        constexpr bool UseConst = !AvoidConst &&
            fitInConstantMemory<FFTSize, float2>();
        if constexpr(UseConst)
        {
            float2* h_twiddles = new float2[FFTSize];
            create_twiddles<-1>(h_twiddles, FFTSize);
            cudaMemcpyToSymbol(fft_twiddles<FFTSize>,
                               h_twiddles,
                              (FFTSize) * sizeof(float2));
        }
    }

    template<unsigned int FFTSize, bool AvoidConst = false>
    void ifft1dCT_plan()
    {
        constexpr bool UseConst = !AvoidConst &&
            fitInConstantMemory<FFTSize, float2>();
        if constexpr(UseConst)
        {
            float2* h_twiddles = new float2[FFTSize];
            create_twiddles<1>(h_twiddles, FFTSize);
            cudaMemcpyToSymbol(ifft_twiddles<FFTSize>,
                               h_twiddles,
                              (FFTSize) * sizeof(float2));
        }
    }

    template<unsigned int FFTSize>
    void conv1dCT_plan()
    {
        constexpr bool useConst =
            fitInConstantMemory<2 * FFTSize, float2>();

        fft1dCT_plan<FFTSize, !useConst>();
        ifft1dCT_plan<FFTSize, !useConst>();
    }
}

#endif
