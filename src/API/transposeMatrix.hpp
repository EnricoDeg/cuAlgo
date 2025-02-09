/*
 * @file transposeMatrix.hpp
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

template <
unsigned int BlockSizeX,
unsigned int BlockSizeY,
typename T>
CUALGO_GLOBAL
void transposeMatrixKernel(const T * CUALGO_RESTRICT idata,
                           T * CUALGO_RESTRICT odata,
                           unsigned int width,
                           unsigned int height)
{

    static constexpr unsigned int block_rows = BlockSizeY / 4;

    CUALGO_SHMEM T tile[BlockSizeY][BlockSizeX+1];
    unsigned int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    unsigned int xIndex = blockIdx_x * BlockSizeX + threadIdx.x;
    unsigned int yIndex = blockIdx_y * BlockSizeY + threadIdx.y;
    unsigned int index_in = xIndex + yIndex * width;
    xIndex = blockIdx_y * BlockSizeX + threadIdx.x;
    yIndex = blockIdx_x * BlockSizeY + threadIdx.y;
    unsigned int index_out = xIndex + yIndex * height;
    for (unsigned int i = 0; i < BlockSizeY; i += block_rows) {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
    __syncthreads();
    for (unsigned int i = 0; i < BlockSizeX; i += block_rows) {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform matrix transposition
    * 
    * @details The input matrix has dimensions {size_x, size_y} and 
    *          the output matrix has dimensions {size_y, size_x}
    * 
    * @param[in]  idata pointer to input matrix to be transposed
    * @param[out] odata pointer to output matrix with result of the transposition
    * @param[in]  size_x contiguous dimension of the input matrix
    * @param[in]  size_y non-contiguous dimension of the input matrix
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    typename T>
    void transposeMatrix(T *idata,
                         T *odata,
                         unsigned int size_x,
                         unsigned int size_y,
                         cudaStream_t stream = 0,
                         bool async = false) {

        static constexpr unsigned int block_rows = BlockSizeY / 4;
        dim3 blocksPerGrid3(size_x / BlockSizeX, size_y / BlockSizeY, 1);
        dim3 threadsPerBlock3(BlockSizeX, block_rows, 1);

        print_kernel_config(threadsPerBlock3, blocksPerGrid3) ;

        TIME(blocksPerGrid3, threadsPerBlock3, 0, stream, async, 
             CUALGO_KERNEL_NAME(transposeMatrixKernel<BlockSizeX, BlockSizeY, T>),
             idata, odata, size_x, size_y);
    }
}
