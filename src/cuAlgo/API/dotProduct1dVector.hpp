/*
 * @file dotProduct1dVector.hpp
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
#include "cuAlgo/internals/gOperationAndReduction1dVector.hpp"

namespace cuAlgo {

/**
 * @brief   Compute dot product of two vectors
 * 
 * @details Sum product of elements of the input vectors and
 *          return a pointer to a scalar
 * 
 * @param[in]  g_idata1 pointer to input vector
 * @param[in]  g_idata2 pointer to input vector
 * @param[out] g_odata pointer to output scalar with result of the L1 norm
 * @param[in]  size  size of the input vector
 * @param[in]  stream CUDA stream where the kernels are launched.
 *                    Default is stream 0 (default stream)
 * @param[in]  async  bool to define if kernels are launched asynchronously
 *                    (without synchronization).
 *                    Default is false (device is synchronized after each kernel launched)
 * 
 * @ingroup algo
 */
    template<
    typename T,
    unsigned int threadsPerBlock,
    unsigned int ItemsPerThread
    >
    void dotProduct1dVector(T            *g_idata1     ,
                            T            *g_idata2     ,
                            T            *g_odata      ,
                            unsigned int  size         ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false) {

        unsigned int blocksPerGrid = size / (ItemsPerThread*threadsPerBlock) +
                                    (size % (ItemsPerThread*threadsPerBlock) > 0);

		gOperationAndReduction1dVector<T, dotProduct_impl, threadsPerBlock, ItemsPerThread>(
                                            g_idata1       ,
		                                    g_idata2       ,
		                                    g_odata        ,
		                                    size           ,
		                                    stream         ,
		                                    async          ,
		                                    blocksPerGrid  ) ;
    }
}
