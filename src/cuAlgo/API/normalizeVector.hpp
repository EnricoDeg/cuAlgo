/*
 * @file normalizeVector.hpp
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
#include "cuAlgo/API/normL1Vector.hpp"

template<typename T>
CUALGO_GLOBAL
void normalizeKernel(T * CUALGO_RESTRICT data,
                     T * CUALGO_RESTRICT norm,
                     unsigned int size) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        data[i] /= (*norm);
        i += gridDim.x * blockDim.x;
    }
}

namespace cuAlgo {

    /**
    * @brief   Normalize a vector
    * 
    * @details Each element of the vector is divided by the L1 norm
    *          of the vector
    * 
    * @param[inout]  idata pointer to input vector which will be normalized
    * @param[in]     size  size of the input vector
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename T>
    void normalizeVector(T *g_idata,
                         unsigned int size,
                         cudaStream_t stream = 0,
                         bool async = false) {

        T * g_odata;
        check_cuda( cudaMalloc(&g_odata, sizeof(T)) );

        normL1Vector<T, BlockSize, ItemsPerThread>(g_idata, g_odata, size, stream, async);

        dim3 threadsPerBlock(BlockSize);
        dim3 blocksPerGrid(div_ceil(size, BlockSize));
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME(blocksPerGrid, threadsPerBlock, 0, stream, async,
             normalizeKernel<T>,
             g_idata, g_odata, size);

        check_cuda( cudaFree ( g_odata ) );
    }
}
