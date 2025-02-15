/*
 * @file gSpMatVecMulCSRAdaptive.hpp
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
#include "cuAlgo/internals/utils.hpp"
#include "cuAlgo/internals/kernelParameters.hpp"

template <
unsigned int NumberNonZerosPerBlock,
unsigned int WarpSize,
typename T>
CUALGO_GLOBAL
void gSpMatVecMulCSRAdaptiveKernel ( const unsigned int * CUALGO_RESTRICT columns,
                                     const unsigned int * CUALGO_RESTRICT row_ptr,
                                     const unsigned int * CUALGO_RESTRICT row_blocks,
                                     const T * CUALGO_RESTRICT values,
                                     const T * CUALGO_RESTRICT x,
                                     T * CUALGO_RESTRICT y,
                                     unsigned int nrows) {

    const unsigned int block_row_begin = row_blocks[blockIdx.x];
    const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
    const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

    CUALGO_SHMEM T cache[NumberNonZerosPerBlock];

    if (block_row_end - block_row_begin > 1) {

        // CSR-Stream case
        const unsigned int i = threadIdx.x;
        const unsigned int block_data_begin = row_ptr[block_row_begin];
        const unsigned int thread_data_begin = block_data_begin + i;

        if (i < nnz)
            cache[i] = values[thread_data_begin] * x[columns[thread_data_begin]];
        __syncthreads ();

        const unsigned int threads_for_reduction = prev_power_of_2(
            blockDim.x / (block_row_end - block_row_begin));

        if (threads_for_reduction > 1) {

            // Reduce all non zeroes of row by multiple thread
            const unsigned int thread_in_block = i % threads_for_reduction;
            const unsigned int local_row = block_row_begin + i / threads_for_reduction;

            T dot = 0;

            if (local_row < block_row_end) {

                const unsigned int local_first_element = row_ptr[local_row]     - row_ptr[block_row_begin];
                const unsigned int local_last_element  = row_ptr[local_row + 1] - row_ptr[block_row_begin];

                for (unsigned int local_element = local_first_element + thread_in_block;
                                  local_element < local_last_element;
                                  local_element += threads_for_reduction) {
                    dot += cache[local_element];
                }
            }
            __syncthreads ();
            cache[i] = dot;

            // Now each row has threads_for_reduction values in cache
            for (unsigned int j = threads_for_reduction / 2; j > 0; j /= 2) {

                // Reduce for each row
                __syncthreads ();

                const bool use_result = thread_in_block < j && i + j < NumberNonZerosPerBlock;
                if (use_result)
                    dot += cache[i + j];
                __syncthreads ();

                if (use_result)
                    cache[i] = dot;
            }

            if (thread_in_block == 0 && local_row < block_row_end)
                y[local_row] = dot;
        } else {

            // Reduce all non zeroes of row by single thread
            unsigned int local_row = block_row_begin + i;
            while (local_row < block_row_end) {

                T dot = 0;

                for (unsigned int j = row_ptr[local_row] - block_data_begin;
                                  j < row_ptr[local_row + 1] - block_data_begin;
                                  j++) {
                    dot += cache[j];
                }

                y[local_row] = dot;
                local_row += NumberNonZerosPerBlock;
            }
        }

    } else {

        const unsigned int row     = block_row_begin;
        const unsigned int warp_id = threadIdx.x / WarpSize;
        const unsigned int lane    = threadIdx.x % WarpSize;

        T dot = 0;

        if (nnz <= 64 || NumberNonZerosPerBlock <= 32) {

            // CSR-Vector case
            if (row < nrows) {

                const unsigned int row_start = row_ptr[row];
                const unsigned int row_end   = row_ptr[row + 1];

                for (unsigned int element = row_start + lane; element < row_end; element += WarpSize)
                    dot += values[element] * x[columns[element]];
            }

            dot = warp_reduce (dot);

            if (lane == 0 && warp_id == 0 && row < nrows)
                y[row] = dot;
        } else {

            // CSR-VectorL case
            if (row < nrows) {

                const unsigned int row_start = row_ptr[row];
                const unsigned int row_end = row_ptr[row + 1];
                for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
                    dot += values[element] * x[columns[element]];
            }

            dot = warp_reduce (dot);

            if (lane == 0)
                cache[warp_id] = dot;
            __syncthreads ();

            if (warp_id == 0) {

                dot = 0.0;

                for (unsigned int element = lane; element < blockDim.x / WarpSize; element += WarpSize)
                    dot += cache[element];

                dot = warp_reduce (dot);

                if (lane == 0 && row < nrows)
                    y[row] = dot;
            }
        }
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform sparse matrix-vector multiplication with an 
    *          adaptive method.
    * 
    * @details The sparse matrix vector multiplication assumes that
    *          the matrix is provided in CSR format.
    *          The adaptive method uses CSR-Vector, CSR-VectorL or
    *          CSR-Stream depending on the local characteristics of
    *          the matrix.
    *          It should be used when the matrix has a low number of 
    *          non zero elements in some rows. In case of high 
    *          number of non zero elements on the all matrix (more 
    *          than 64), the function gSpMatVecMulCSRVector() 
    *          should be used.
    * 
    * @param[in]  columns    An integer array of column positions
    *                        where the matrix value is non zero.
    * @param[in]  row_ptr    Array of locations in the columns array
    *                        where a new row starts.
    * @param[in]  row_blocks An integer array with the number of rows
    *                        for each block. The function getRowBlocks()
    *                        can provide the array.
    * @param[in]  values     An array of non zeros values of the matrix.
    * @param[in]  x          The vector array that multiplies the matrix.
    * @param[out] y          The vector array result of the multiplication.
    * @param[in]  nrows      Number of rows in the matrix.
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <
    unsigned int NumberNonZerosPerBlock,
    unsigned int WarpSize,
    typename T>
    void gSpMatVecMulCSRAdaptive(unsigned int *columns,
                                 unsigned int *row_ptr,
                                 unsigned int *row_blocks,
                                 T *values,
                                 T *x,
                                 T *y,
                                 unsigned int nrows,
                                 unsigned int blocks_count,
                                 cudaStream_t stream = 0,
                                 bool async = false) {

        dim3 threadsPerBlock(NumberNonZerosPerBlock);
        dim3 blocksPerGrid(blocks_count);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gSpMatVecMulCSRAdaptiveKernel<NumberNonZerosPerBlock, WarpSize, T>),
              columns, row_ptr, row_blocks, values, x, y, nrows );
    }
}
