/*
 * @file utils.hpp
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
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
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/checkError.hpp"

CUALGO_HOST_DEVICE int div_ceil(int numerator, int denominator)
{
    return (numerator % denominator != 0) ?
           (numerator / denominator+ 1  ) :
           (numerator / denominator     ) ;
}

template<unsigned int N>
constexpr bool is_power_of_two()
{
    return N && ((N & (N - 1)) == 0);
}

template<unsigned int N>
constexpr bool is_power_of_four()
{
    return (N & 0x55555555u) && is_power_of_two<N>();
}

template <typename T>
size_t getSmem(size_t K) {

    int dev_id = 0;
    check_cuda( cudaGetDevice(&dev_id) );

    cudaDeviceProp dev_prop;
    check_cuda( cudaGetDeviceProperties(&dev_prop, dev_id) );

    size_t smem_max_size = K * sizeof(T);

    if ( dev_prop.sharedMemPerMultiprocessor < smem_max_size) {
        std::cout << "shared memory request too large" << std::endl;
        exit(EXIT_FAILURE);
    }

    return smem_max_size;
}

template<unsigned int WarpSize>
CUALGO_DEVICE int warp_reduce(int val) {
    for (size_t offset = WarpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    return val;
}

CUALGO_DEVICE unsigned int prev_power_of_2 (unsigned int n) {
    while (n & n - 1)
        n = n & n - 1;
    return n;
}

void print_kernel_config(dim3 threadsPerBlock, dim3 blocksPerGrid) {

#ifdef CUALGO_VERBOSE
    std::cout << "threadsPerBlock = " << threadsPerBlock.x << ", " <<
                                         threadsPerBlock.y << ", " <<
                                         threadsPerBlock.z << std::endl;
    std::cout << "blocksPerGrid   = " << blocksPerGrid.x   << ", " <<
                                         blocksPerGrid.y   << ", " <<
                                         blocksPerGrid.z   << std::endl;
#endif
}

#ifdef CUALGO_VERBOSE

#define TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async, func, args ...)               \
  do {                                                                                           \
    auto start = high_resolution_clock::now();                                                   \
    func<<< blocksPerGrid, threadsPerBlock, shmem, stream >>>(args);                             \
    if (!async) {                                                                                \
        check_cuda( cudaPeekAtLastError() ) ;                                                    \
        check_cuda( cudaStreamSynchronize(stream) );                                             \
    }                                                                                            \
    auto stop = high_resolution_clock::now();                                                    \
    auto duration = duration_cast<microseconds>(stop - start);                                   \
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl; \
  } while(0)

#else

#define TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async, func, args ...) \
  do {                                                                             \
    func<<< blocksPerGrid, threadsPerBlock, shmem, stream >>>(args);               \
    if (!async)                                                                    \
        check_cuda( cudaStreamSynchronize(stream) );                               \
  } while(0)

#endif

#endif
