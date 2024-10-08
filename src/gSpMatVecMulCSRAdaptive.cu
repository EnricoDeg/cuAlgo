/*
 * @file gSpMatVecMulCSRAdaptive.cu
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
#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

template <typename T>
__global__ void gSpMatVecMulCSRAdaptiveKernel ( const unsigned int * __restrict__ columns   ,
                                                const unsigned int * __restrict__ row_ptr   ,
                                                const unsigned int * __restrict__ row_blocks,
                                                const T            * __restrict__ values    ,
                                                const T            * __restrict__ x         ,
                                                      T            * __restrict__ y         ,
                                                      unsigned int                nrows     ) {

	const unsigned int block_row_begin = row_blocks[blockIdx.x];
	const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
	const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

	__shared__ T cache[NNZ_PER_WG];

	if (block_row_end - block_row_begin > 1) {

		// CSR-Stream case
		const unsigned int i = threadIdx.x;
		const unsigned int block_data_begin = row_ptr[block_row_begin];
		const unsigned int thread_data_begin = block_data_begin + i;

		if (i < nnz)
			cache[i] = values[thread_data_begin] * x[columns[thread_data_begin]];
		__syncthreads ();

		const unsigned int threads_for_reduction = prev_power_of_2 (blockDim.x / (block_row_end - block_row_begin));

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

				const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;

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
				local_row += NNZ_PER_WG;
			}
		}

	} else {

		const unsigned int row     = block_row_begin;
		const unsigned int warp_id = threadIdx.x / WARP_SIZE;
		const unsigned int lane    = threadIdx.x % WARP_SIZE;

		T dot = 0;

		if (nnz <= 64 || NNZ_PER_WG <= 32) {

			// CSR-Vector case
			if (row < nrows) {

				const unsigned int row_start = row_ptr[row];
				const unsigned int row_end   = row_ptr[row + 1];

				for (unsigned int element = row_start + lane; element < row_end; element += WARP_SIZE)
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

				for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
					dot += cache[element];

				dot = warp_reduce (dot);

				if (lane == 0 && row < nrows)
					y[row] = dot;
			}
		}
	}
}

template <typename T>
void gSpMatVecMulCSRAdaptive(unsigned int *columns     ,
                             unsigned int *row_ptr     ,
                             unsigned int *row_blocks  ,
                                      T   *values      ,
                                      T   *x           ,
                                      T   *y           ,
                             unsigned int  nrows       ,
                             unsigned int  blocks_count,
                             cudaStream_t  stream      ,
                             bool          async       ) {

	dim3 threadsPerBlock(NNZ_PER_WG);
	dim3 blocksPerGrid(blocks_count);
	print_kernel_config(threadsPerBlock, blocksPerGrid);

	TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
	      gSpMatVecMulCSRAdaptiveKernel<T>,
	      columns, row_ptr, row_blocks, values, x, y, nrows );
}

namespace cuAlgo {

	void gSpMatVecMulCSRAdaptiveInt(unsigned int *columns     ,
	                                unsigned int *row_ptr     ,
	                                unsigned int *row_blocks  ,
	                                         int *values      ,
	                                         int *x           ,
	                                         int *y           ,
	                                unsigned int  nrows       ,
	                                unsigned int  blocks_count,
	                                cudaStream_t  stream      ,
	                                bool          async       )
	{

		gSpMatVecMulCSRAdaptive<int>(columns, row_ptr, row_blocks , values,
		                             x, y, nrows, blocks_count, stream, async);
	}

	void gSpMatVecMulCSRAdaptiveFloat(unsigned int   *columns     ,
	                                  unsigned int   *row_ptr     ,
	                                  unsigned int   *row_blocks  ,
	                                           float *values      ,
	                                           float *x           ,
	                                           float *y           ,
	                                  unsigned int    nrows       ,
	                                  unsigned int    blocks_count,
	                                  cudaStream_t    stream      ,
	                                  bool            async       )
	{

		gSpMatVecMulCSRAdaptive<float>(columns, row_ptr, row_blocks , values,
		                               x, y, nrows, blocks_count, stream, async);
	}

	void gSpMatVecMulCSRAdaptiveDouble(unsigned int    *columns     ,
	                                   unsigned int    *row_ptr     ,
	                                   unsigned int    *row_blocks  ,
	                                            double *values      ,
	                                            double *x           ,
	                                            double *y           ,
	                                   unsigned int     nrows       ,
	                                   unsigned int     blocks_count,
	                                   cudaStream_t     stream      ,
	                                   bool             async       )
	{

		gSpMatVecMulCSRAdaptive<double>(columns, row_ptr, row_blocks , values,
		                                x, y, nrows, blocks_count, stream, async);
	}
}
