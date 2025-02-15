/*
 * @file exclusiveScan1dVector.hpp
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
#include "cuAlgo/internals/templateShMem.hpp"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

template <typename T>
CUALGO_GLOBAL
void exclusiveScan1dKernelBlock(const T * CUALGO_RESTRICT g_idata,
                                T * CUALGO_RESTRICT g_odata,
                                unsigned int size) {

    // use dynamic shared memory
    // needed for template
    SharedMemory<T> smem;
    T * temp = smem.getPointer();

    unsigned int thid = threadIdx.x;
    unsigned int offset = 1;
    unsigned int ai = thid;
    unsigned int bi = thid + (size / 2);
    unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    for (unsigned int d = size>>1 ; d > 0 ; d >>=1) {

        __syncthreads();
        if (thid < d) {

            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
        temp[ size - 1 + CONFLICT_FREE_OFFSET(size - 1) ] = 0;

    for (unsigned int d = 1; d < size; d *= 2) {

        offset >>= 1;
        __syncthreads();
        if (thid < d) {

            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}

template <typename T>
CUALGO_GLOBAL
void exclusiveScan1dKernelMultiBlock(const T * CUALGO_RESTRICT g_idata,
                                     T * CUALGO_RESTRICT g_odata,
                                     T * CUALGO_RESTRICT sums,
                                     unsigned int size) {

    // use dynamic shared memory
    // needed for template
    SharedMemory<T> smem;
    T * temp = smem.getPointer();

    unsigned int blockID = blockIdx.x;
    unsigned int threadID = threadIdx.x;
    unsigned int blockOffset = blockID * size;

    unsigned int ai = threadID;
    unsigned int bi = threadID + (size / 2);
    unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
    temp[bi + bankOffsetB] = g_idata[blockOffset + bi];

    // build sum in place up the tree
    unsigned int offset = 1;
    for (unsigned int d = size >> 1; d > 0; d >>= 1) {

        __syncthreads();
        if (threadID < d) {

            unsigned int ai = offset * (2 * threadID + 1) - 1;
            unsigned int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) { 
        sums[blockID] = temp[size - 1 + CONFLICT_FREE_OFFSET(size - 1)];
        temp[size - 1 + CONFLICT_FREE_OFFSET(size - 1)] = 0;
    }

    // traverse down tree & build scan
    for (unsigned int d = 1; d < size; d *= 2) {

        offset >>= 1;
        __syncthreads();
        if (threadID < d) {

            unsigned int ai = offset * (2 * threadID + 1) - 1;
            unsigned int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    g_odata[blockOffset + ai] = temp[ai + bankOffsetA];
    g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
}

template <typename T>
CUALGO_GLOBAL
void add(T * CUALGO_RESTRICT output,
         unsigned int length,
         T * n) {

    unsigned int blockID = blockIdx.x;
    unsigned int threadID = threadIdx.x;
    unsigned int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}

namespace cuAlgo {

    /**
    * @brief   Perform exclusive scan or prefix sum on a vector
    * 
    * @details The input and output arrays are expected to be multiple of 
    *          1024. If not, they should be padded before calling the 
    *          function.
    * 
    * @param[in]  g_idata input array of size {size}
    * @param[out] g_odata output array of size {size}
    * @param[in]  size size of input and output arrays
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
    typename T>
    void exclusiveScan1dVector(T *g_idata,
                               T *g_odata,
                               unsigned int size,
                               cudaStream_t stream = 0,
                               bool async = false) {

        unsigned int blocks = size / BlockSize;
        T *d_sums, *d_incr;
        check_cuda( cudaMalloc(&d_sums, blocks * sizeof(T)) );
        check_cuda( cudaMalloc(&d_incr, blocks * sizeof(T)) );

        // Multi blocks
        {

            dim3 threadsPerBlock(BlockSize / 2);
            dim3 blocksPerGrid(div_ceil(size, BlockSize));
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            unsigned int shmem = BlockSize * sizeof(T);

            TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async,
                 exclusiveScan1dKernelMultiBlock<T>,
                 g_idata, g_odata, d_sums, BlockSize);
        }

        // Multi block (recursion) or single block
        const unsigned int sumsArrThreadsNeeded = (blocks + 1) / 2;
        if (sumsArrThreadsNeeded > BlockSize / 2) {

            exclusiveScan1dVector<BlockSize, T>(d_sums, d_incr, blocks, stream, async);
        } else {

            dim3 threadsPerBlock((blocks + 1) / 2);
            dim3 blocksPerGrid(1);
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            unsigned int shmem = (blocks + 1) / 2 * sizeof(T);

            TIME( blocksPerGrid, threadsPerBlock, shmem, stream, async,
                  exclusiveScan1dKernelBlock<T>,
                  d_sums, d_incr, blocks );
        }

        // Final step
        {

            dim3 threadsPerBlock(BlockSize);
            dim3 blocksPerGrid(blocks);
            print_kernel_config(threadsPerBlock, blocksPerGrid);

            TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
                  add<T>,
                  g_odata, BlockSize, d_incr );
        }

        check_cuda( cudaFree ( d_sums ) );
        check_cuda( cudaFree ( d_incr ) );
    }
}
