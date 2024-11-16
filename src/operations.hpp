/*
 * @file operations.hpp
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

#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cuda.h>

template <typename T>
class reductionSum_impl {

	public:
	__device__ inline T globalMemory(T * __restrict__ a, T * __restrict__ b ) {
		return *a + *b;
	}
	__device__ inline void loadSharedMemory(volatile T * result, T * __restrict__ a) {
		*result += *a;
	}
	__device__ inline void sharedMemory(volatile T * result, volatile T * data) {
		*result += *data;
	}
};

template <typename T>
class reductionProd_impl {

	public:
	__device__ inline T globalMemory(T * __restrict__ a, T * __restrict__ b ) {
		return *a * *b;
	}
	__device__ inline void loadSharedMemory(volatile T * result, T * __restrict__ a) {
		*result *= *a;
	}
	__device__ inline void sharedMemory(volatile T * result, volatile T * data) {
		*result *= *data;
	}
};

template <typename T>
class normL1_impl {

	public:
	__device__ inline T globalMemory(T * __restrict__ a, T * __restrict__ b ) {
		return abs(*a) + abs(*b);
	}
	__device__ inline void loadSharedMemory(volatile T * result, T * __restrict__ a) {
		*result += *a;
	}
	__device__ inline void sharedMemory(volatile T * result, volatile T * data) {
		*result += *data;
	}
};

template <typename T>
class normL2_impl {

	public:
	__device__ inline T globalMemory(T * __restrict__ a, T * __restrict__ b ) {
		return (*a) * (*a) + (*b) * (*b);
	}
	__device__ inline void loadSharedMemory(volatile T * result, T * __restrict__ a) {
		*result += *a;
	}
	__device__ inline void sharedMemory(volatile T * result, volatile T * data) {
		*result += *data;
	}
};

template <typename T>
class normLInf_impl {

	public:
	__device__ inline T globalMemory(T * __restrict__ a, T * __restrict__ b ) {
		return max(abs(*a), abs(*b));
	}
	__device__ inline void loadSharedMemory(volatile T * result, T * __restrict__ a) {
		*result = max(*result, *a);
	}
	__device__ inline void sharedMemory(volatile T * result, volatile T * data) {
		*result = max(*result, *data);
	}
};

#endif