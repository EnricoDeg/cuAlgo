/*
 * @file convolution1dMatrix.hpp
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
#include "internals/gConvolutionCorrelation1dMatrix.hpp"

namespace cuAlgo {

    /**
    * @brief   Perform 1D convolution on the input matrices.
    * 
    * @details The convolution is done on the fast dimension of the input
    *          matrices R and V. This means that each convolution in the 
    *          slow dimension is independent.
    *          It can be used to convolve two groups of signals in a single kernel.
    *          The signals are expected to be in the frequency domain.
    *          The input matrices are expected to have the half complex memory
    *          layout. This means that this is a convolution on real signals.
    * 
    * @param[in]  R      pointer to the first input matrix for the convolution.
    *                    The signals are assumed to be in the frequency domain already.
    *                    The matrix has dimensions {N,K}.
    * @param[in]  V      pointer to the second input matrix for the convolution.
    *                    The signals are assumed to be in the frequency domain already.
    *                    The matrix has dimensions {N,K}.
    * @param[out] C      pointer to the output matrix with results of the convolution.
    *                    The signals are still in the frequency domain.
    *                    The matrix has dimension {N,K}.
    * @param[in]  N      contiguous dimension of the input matrix
    * @param[in]  K      non-contiguous dimension of the input matrix
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int ItemsPerThread,
    typename T
    >
    void convolution1dMatrix(T *R,
                             T *V,
                             T *C,
                             unsigned int N,
                             unsigned int K,
                             cudaStream_t stream = 0,
                             bool async = false) {

        gConvolutionCorrelation1dMatrix<BlockSizeX,
                                        BlockSizeY,
                                        ItemsPerThread,
                                        T,
                                        convolution_impl>(R, V, C, N, K, stream, async);
    }
}
