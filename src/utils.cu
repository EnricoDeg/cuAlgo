/*
 * @file utils.cu
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
#include "utils.hpp"

__device__ __host__ int div_ceil(int numerator, int denominator)
{

	return (numerator % denominator != 0) ?
	       (numerator / denominator+ 1  ) :
	       (numerator / denominator     ) ;
}

__device__ int warp_reduce(int val) {

	for (size_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
	return val;
}

// Compute closest number to n which is a power of 2.
// The result is always less or equal to n.
__device__ unsigned int prev_power_of_2 (unsigned int n) {

	while (n & n - 1)
		n = n & n - 1;
	return n;
}