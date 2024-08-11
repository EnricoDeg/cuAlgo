/*
 * @file utils.hpp
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
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include "checkError.hpp"

__device__ __host__ int div_ceil(int numerator, int denominator);

template <typename T>
size_t getSmem(size_t K) {

	int dev_id = 0;
	check_cuda( cudaGetDevice(&dev_id) );

	cudaDeviceProp dev_prop;
	check_cuda( cudaGetDeviceProperties(&dev_prop, dev_id) );

	size_t smem_max_size = K * sizeof(T);

	if ( dev_prop.sharedMemPerMultiprocessor < smem_max_size) {

		std::cout << "shared memory request too large" << std::endl;
		exit(EXIT_FAILURE);
	}

	return smem_max_size;
}

#endif
