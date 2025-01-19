/*
 * @file gOperationAndReduction1dVector.hpp
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

#ifndef GOPERATIONANDREDUCTION1DVECTOR_H
#define GOPERATIONANDREDUCTION1DVECTOR_H

#include <cuda.h>
#include "internals/operations.hpp"
#include "internals/templateShMem.hpp"
#include "internals/utils.hpp"
#include "internals/reductionShMem.hpp"

template <
unsigned int blockSize,
unsigned int ItemsPerThread,
typename T,
template<typename> class op_t
>
CUALGO_GLOBAL
void operationAndReduction1dKernel(T * CUALGO_RESTRICT g_idata1,
                                   T * CUALGO_RESTRICT g_idata2,
                                   T * CUALGO_RESTRICT g_odata ,
                                   unsigned int        n       ,
                                   op_t<T>             Op      ) {

    // use dynamic shared memory
    // needed for template
    SharedMemory<T> smem;
    T * sdata = smem.getPointer();

    // parameters
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = blockDim.x*gridDim.x;

    // load multiple elements to shared memory
    sdata[tid] = 0;
    T a = 0;

    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        if (i < n) {
            a = Op.globalMemory( &a , &g_idata1[i], &g_idata2[i] );
        }
        i += gridSize;
    }
    Op.loadSharedMemory(&sdata[tid], &a);
    __syncthreads();

    // do reduction in shared mem
    blockReduceShMemUnroll<blockSize, T, op_t>(sdata, tid, Op);

    // write result for this block to global mem
    if (tid == 0)
        Op.atomic(&g_odata[0], sdata[0]);
}

template<
typename T,
template<typename> class op_t,
unsigned int threadsPerBlock,
unsigned int ItemsPerThread
>
void gOperationAndReduction1dVector(T            * g_idata1     ,
                                    T            * g_idata2     ,
                                    T            * d_buffer     ,
                                    unsigned int   size         ,
                                    cudaStream_t   stream       ,
                                    bool           async        ,
                                    unsigned int   blocksPerGrid) {

    unsigned int shmem = threadsPerBlock*sizeof(T);
    op_t<T> op;
    dim3 blocksPerGrid3(blocksPerGrid, 1, 1);
    dim3 threadsPerBlock3(threadsPerBlock, 1, 1);
    print_kernel_config(threadsPerBlock3, blocksPerGrid3);

    TIME(blocksPerGrid3, threadsPerBlock3, shmem, stream, async,
         CUALGO_KERNEL_NAME(operationAndReduction1dKernel<threadsPerBlock, ItemsPerThread, T, op_t>),
         g_idata1, g_idata2, d_buffer, size, op);
}

#endif