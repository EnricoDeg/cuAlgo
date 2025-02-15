/*
 * @file fftshift1dVector.hpp
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
#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"

template <typename T>
CUALGO_GLOBAL
void fftshiftVectorKernelEven(T * CUALGO_RESTRICT idata,
                              T * CUALGO_RESTRICT odata,
                              unsigned int size)
{

    const unsigned int tid_in = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_in < size) {

        const unsigned int tid_out = tid_in < size / 2 ?
                                     tid_in + size / 2 :
                                     tid_in - size / 2 ;
        odata[tid_out] = idata[tid_in];
    }
}

template <typename T>
CUALGO_GLOBAL
void fftshiftVectorKernelOdd(T * CUALGO_RESTRICT idata,
                             T * CUALGO_RESTRICT odata,
                             unsigned int size)
{

    const unsigned int tid_in = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_in < size) {

        const unsigned int tid_out = tid_in <= ( size - 1 ) / 2 ?
                                     tid_in +  ( size - 1 ) / 2 :
                                     tid_in -  ( size + 1 ) / 2 ;
        odata[tid_out] = idata[tid_in];
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform fftshift on a vector
    * 
    * @details The input vector has dimension {size} and 
    *          the output vector has dimension {size}
    * 
    * @param[in]  idata pointer to input vector to be shifted
    * @param[out] odata pointer to output vector with result of the fftshift
    * @param[in]  size  contiguous dimension of the input and output vectors
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
    void fftshift1dVector(T *idata ,
                          T *odata ,
                          unsigned int size,
                          cudaStream_t stream = 0,
                          bool async = false) {

        dim3 blocksPerGrid3(size / BlockSize, 1, 1);
        dim3 threadsPerBlock3(BlockSize, 1, 1);

        print_kernel_config(threadsPerBlock3, blocksPerGrid3) ;

        if (size % 2 == 0) {

            TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                 fftshiftVectorKernelEven<T>,
                 idata, odata, size);
        } else {

            TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
                 fftshiftVectorKernelOdd<T>,
                 idata, odata, size);
        }
    }
}
