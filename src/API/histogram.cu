/*
 * @file histogram.cu
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

#include "cuAlgo.h"
#include "internals/utils.hpp"
#include "internals/kernelParameters.hpp"

template<typename T, unsigned int BinSize>
__global__ void histogram_kernel(T * __restrict__ data,
                                 unsigned int size,
                                 unsigned int * __restrict__ histo) {

    __shared__ unsigned int temp[BinSize];
    temp[threadIdx.x] = 0;
    __syncthreads();

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int offset = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd( &temp[data[i]], 1);
        i += offset;
    }
    __syncthreads();

    atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

namespace cuAlgo {

    template<typename T, unsigned int BinSize>
    void histogram(T *data,
                   unsigned int size,
                   unsigned int *histo,
                   cudaStream_t  stream,
                   bool          async ) {

        unsigned int tpb = THREADS_PER_BLOCK > BinSize ? BinSize : THREADS_PER_BLOCK;
        dim3 threadsPerBlock(tpb);
        dim3 blocksPerGrid(div_ceil(size, tpb));
        print_kernel_config(threadsPerBlock, blocksPerGrid);


        TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
            histogram_kernel<T COMMA BinSize>,
            data, size, histo);
    }

    template void histogram<unsigned char, 256>(unsigned char *, unsigned int, unsigned int*, cudaStream_t, bool);
}
