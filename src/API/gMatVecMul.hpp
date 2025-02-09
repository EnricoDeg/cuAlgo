/*
 * @file gMatVecMul.hpp
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
#include "internals/templateShMem.hpp"
#include "internals/kernelParameters.hpp"

__device__ int div_ceil_device(int numerator, int denominator)
{

    return (numerator % denominator != 0) ?
           (numerator / denominator+ 1  ) :
           (numerator / denominator     ) ;
}

template <
unsigned int BlockSize,
unsigned int WarpSize,
typename T>
CUALGO_GLOBAL
void gMatVecMulKernel(const T * CUALGO_RESTRICT A,
                      const T * CUALGO_RESTRICT B,
                      T * CUALGO_RESTRICT C,
                      unsigned int N,
                      unsigned int K) {

    static constexpr unsigned int WarpsPerBlock = BlockSize / WarpSize;

    // use dynamic shared memory
    // needed for template
    SharedMemory<T> smem;
    T * A_smem = smem.getPointer();

    unsigned int A_smem_iters = div_ceil_device(K, BlockSize);

    CUALGO_UNROLL
    for (unsigned int i = 0; i < A_smem_iters; ++i) {
        unsigned int idx = i * BlockSize + threadIdx.x;
        A_smem[idx] = A[idx];
    }
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / WarpSize;
    const unsigned int warp_col = blockIdx.x * WarpsPerBlock + warp_id;
    if (warp_col >= N)
        return;

    const unsigned int K_iters = div_ceil_device(K, WarpSize);
    const unsigned int lane_id = threadIdx.x % WarpSize;

    T tmp = 0.0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K_iters; ++i) {
        const unsigned int A_idx = i * WarpSize + lane_id;
        const unsigned int B_idx = i * WarpSize + lane_id + warp_col * K;
        tmp += A_smem[A_idx] * B[B_idx];
    }

    const unsigned int mask = 0xffffffff;
    CUALGO_UNROLL
    for (unsigned int i = WarpSize / 2; i >= 1; i /= 2)
        tmp += __shfl_xor_sync(mask, tmp, i);

    if (lane_id == 0)
        C[warp_col] = tmp;
}

template<
unsigned int BlockSize,
unsigned int WarpSize,
unsigned int ColsPerWarp,
typename T>
CUALGO_GLOBAL
void gMatVecMulKernel1(const T * CUALGO_RESTRICT A,
                       const T * CUALGO_RESTRICT B,
                       T * CUALGO_RESTRICT C,
                       unsigned int N,
                       unsigned int K) {

    static constexpr unsigned int WarpsPerBlock = BlockSize / WarpSize;
    static constexpr unsigned int ColsPerBlock = ColsPerWarp * WarpsPerBlock;
    static constexpr unsigned int GroupSize = WarpSize / ColsPerWarp;

    // use dynamic shared memory
    // needed for template
    SharedMemory<T> smem;
    T * A_smem = smem.getPointer();

    unsigned int A_smem_iters = div_ceil_device(K, BlockSize);

    CUALGO_UNROLL
    for (unsigned int i = 0; i < A_smem_iters; ++i) {
        unsigned int idx = i * BlockSize + threadIdx.x;
        A_smem[idx] = A[idx];
    }
    __syncthreads();

    const unsigned int group_id  = threadIdx.x / GroupSize;
    const unsigned int group_col = blockIdx.x * ColsPerBlock + group_id;
    if (group_col >= N)
        return;

    const unsigned int K_iters = div_ceil_device(K, GroupSize);
    const unsigned int group_lane_id = threadIdx.x % GroupSize;

    T tmp = 0.0;
    CUALGO_UNROLL
    for (unsigned int i = 0; i < K_iters; ++i) {
        const unsigned int A_idx = i * GroupSize + group_lane_id;
        const unsigned int B_idx = i * GroupSize + group_lane_id + group_col * K;
        tmp += A_smem[A_idx] * B[B_idx];
    }

    constexpr unsigned int mask = 0xffffffff;
    CUALGO_UNROLL
    for (unsigned int i = GroupSize / 2; i >= 1; i /= 2)
        tmp += __shfl_xor_sync(mask, tmp, i);

    if (group_lane_id == 0)
        C[group_col] = tmp;
}

namespace cuAlgo {

    /**
    * @brief   Perform matrix-vector multiplication.
    * 
    * @details The vector A is multiplied with matrix B and the result is stored 
    *          in vector C.
    *          B * A = C
    * 
    * @param[in]  A pointer to the input vector.
    *               The vector has dimensions {K}.
    * @param[in]  B pointer to the input matrix.
    *               The matrix has dimensions {K,N}.
    * @param[out] C pointer to the output vector.
    *               The vector has dimensions {N}.
    * @param[in]  N contiguous dimension of the input matrix
    * @param[in]  K non-contiguous dimension of the input matrix
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
    unsigned int WarpSize,
    unsigned int ColsPerWarp,
    typename T>
    void gMatVecMul(const T *A,
                    const T *B,
                    T *C,
                    unsigned int  N,
                    unsigned int  K,
                    cudaStream_t stream = 0,
                    bool async = false) {

        static constexpr unsigned int WarpsPerBlock = BlockSize / WarpSize;
        dim3 threadsPerBlock(BlockSize);
        dim3 blocksPerGrid(div_ceil(N, WarpsPerBlock));
        print_kernel_config(threadsPerBlock, blocksPerGrid);
        unsigned int smem = getSmem<T>(K);

        TIME( blocksPerGrid, threadsPerBlock, smem, stream, async, 
              CUALGO_KERNEL_NAME(gMatVecMulKernel1<BlockSize, WarpSize, ColsPerWarp, T>),
              A, B, C, N, K );
    }
}
