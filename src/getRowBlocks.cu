/*
 * @file getRowBlocks.cu
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
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

int * getRowBlocks( const int * row_ptr     ,
                          int   nrows       ,
                          int * blocks_count) {

	auto start = high_resolution_clock::now();

	// Part 1: compute size of row_blocks
	int last_i = 0;
	int current_wg = 1;
	unsigned int nnz_sum = 0;
	for (int i = 1; i <= nrows; i++) {

		nnz_sum += row_ptr[i] - row_ptr[i - 1];

		if (nnz_sum == NNZ_PER_WG) {
			last_i = i;

			current_wg++;
			nnz_sum = 0;
		} else if (nnz_sum > NNZ_PER_WG) {

			if (i - last_i > 1) {

				current_wg++;
				i--;
			} else {
				current_wg++;
			}

			last_i = i;
			nnz_sum = 0;
		} else if (i - last_i > NNZ_PER_WG) {

			last_i = i;
			current_wg++;
			nnz_sum = 0;
		}
	}

	// Part 2: Create and fill row_blocks
	int * row_blocks = (int *)malloc((current_wg + 1) * sizeof(int));

	row_blocks[0] = 0;

	last_i = 0;
	current_wg = 1;
	nnz_sum = 0;
	for (int i = 1; i <= nrows; i++) {

		nnz_sum += row_ptr[i] - row_ptr[i - 1];

		if (nnz_sum == NNZ_PER_WG) {

			last_i = i;

			row_blocks[current_wg] = i;
			current_wg++;
			nnz_sum = 0;
		} else if (nnz_sum > NNZ_PER_WG) {

			if (i - last_i > 1) {

				row_blocks[current_wg] = i - 1;
				current_wg++;
				i--;
			} else {

				row_blocks[current_wg] = i;
				current_wg++;
			}

			last_i = i;
			nnz_sum = 0;
		} else if (i - last_i > NNZ_PER_WG) {

			last_i = i;
			row_blocks[current_wg] = i;
			current_wg++;
			nnz_sum = 0;
		}
	}

	row_blocks[current_wg] = nrows;

	*blocks_count = current_wg;

	int * d_row_blocks;
	check_cuda( cudaMalloc(&d_row_blocks, (current_wg + 1) * sizeof (int)) );
	check_cuda( cudaMemcpy( d_row_blocks, row_blocks, sizeof (int) * (current_wg + 1), cudaMemcpyHostToDevice) );
	free(row_blocks);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by getRowBlocks function: " << duration.count() << " microseconds" << std::endl;

	return d_row_blocks;
}
