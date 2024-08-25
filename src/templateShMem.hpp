/*
 * @file templateShMem.hpp
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

#ifndef TEMPLATESHMEM_HPP
#define TEMPLATESHMEM_HPP

#include <cuda.h>

/** @brief Wrapper class for templatized dynamic shared memory arrays.
  * 
  * This struct uses template specialization on the type \a T to declare
  * a differently named dynamic shared memory array for each type
  * (\code extern __shared__ T s_type[] \endcode).
  * 
  * Currently there are specializations for the following types:
  * \c int, \c uint, \c char, \c uchar, \c short, \c ushort, \c long, 
  * \c unsigned long, \c bool, \c float, and \c double. One can also specialize it
  * for user defined types.
  */
template <typename T>
struct SharedMemory
{

	__device__ T* getPointer() {
		extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
		Error_UnsupportedType();
		return (T*)0;
	}
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double

template <>
struct SharedMemory <int> {
	__device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }      
};

template <>
struct SharedMemory <unsigned int> {
	__device__ unsigned int* getPointer() { extern __shared__ unsigned int s_uint[]; return s_uint; }    
};

template <>
struct SharedMemory <char> {
	__device__ char* getPointer() { extern __shared__ char s_char[]; return s_char; }    
};

template <>
struct SharedMemory <unsigned char> {
	__device__ unsigned char* getPointer() { extern __shared__ unsigned char s_uchar[]; return s_uchar; }    
};

template <>
struct SharedMemory <short> {
	__device__ short* getPointer() { extern __shared__ short s_short[]; return s_short; }    
};

template <>
struct SharedMemory <unsigned short> {
	__device__ unsigned short* getPointer() { extern __shared__ unsigned short s_ushort[]; return s_ushort; }    
};

template <>
struct SharedMemory <long> {
	__device__ long* getPointer() { extern __shared__ long s_long[]; return s_long; }    
};

template <>
struct SharedMemory <unsigned long> {
	__device__ unsigned long* getPointer() { extern __shared__ unsigned long s_ulong[]; return s_ulong; }    
};

template <>
struct SharedMemory <long long> {
	__device__ long long* getPointer() { extern __shared__ long long s_longlong[]; return s_longlong; }    
};

template <>
struct SharedMemory <unsigned long long> {
	__device__ unsigned long long* getPointer() { extern __shared__ unsigned long long s_ulonglong[]; return s_ulonglong; }    
};

template <>
struct SharedMemory <bool> {
	__device__ bool* getPointer() { extern __shared__ bool s_bool[]; return s_bool; }    
};

template <>
struct SharedMemory <float> {
	__device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }    
};

template <>
struct SharedMemory <double> {
	__device__ double* getPointer() { extern __shared__ double s_double[]; return s_double; }    
};

template <>
struct SharedMemory <uchar4> {
	__device__ uchar4* getPointer() { extern __shared__ uchar4 s_uchar4[]; return s_uchar4; }    
};

#endif
