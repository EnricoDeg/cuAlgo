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
#include "thrust/complex.h"
#include "cuAlgo/internals/definitions.hpp"

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
    CUALGO_DEVICE T* getPointer() {
        // Ensure that we won't compile any un-specialized types
        extern CUALGO_DEVICE void Error_UnsupportedType();
        Error_UnsupportedType();
        return (T*)0;
    }
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double

template <>
struct SharedMemory <int> {
    CUALGO_DEVICE int* getPointer() {
        extern CUALGO_SHMEM int s_int[];
        return s_int;
    }
};

template <>
struct SharedMemory <unsigned int> {
    CUALGO_DEVICE unsigned int* getPointer() {
        extern CUALGO_SHMEM unsigned int s_uint[];
        return s_uint;
    }
};

template <>
struct SharedMemory <char> {
    CUALGO_DEVICE char* getPointer() {
        extern CUALGO_SHMEM char s_char[];
        return s_char;
    }
};

template <>
struct SharedMemory <unsigned char> {
    CUALGO_DEVICE unsigned char* getPointer() {
        extern CUALGO_SHMEM unsigned char s_uchar[];
        return s_uchar;
    }
};

template <>
struct SharedMemory <short> {
    CUALGO_DEVICE short* getPointer() {
        extern CUALGO_SHMEM short s_short[];
        return s_short;
    }
};

template <>
struct SharedMemory <unsigned short> {
    CUALGO_DEVICE unsigned short* getPointer() {
        extern CUALGO_SHMEM unsigned short s_ushort[];
        return s_ushort;
    }
};

template <>
struct SharedMemory <long> {
    CUALGO_DEVICE long* getPointer() {
        extern CUALGO_SHMEM long s_long[];
        return s_long;
    }
};

template <>
struct SharedMemory <unsigned long> {
    CUALGO_DEVICE unsigned long* getPointer() {
        extern CUALGO_SHMEM unsigned long s_ulong[];
        return s_ulong;
    }
};

template <>
struct SharedMemory <long long> {
    CUALGO_DEVICE long long* getPointer() {
        extern CUALGO_SHMEM long long s_longlong[];
        return s_longlong;
    }
};

template <>
struct SharedMemory <unsigned long long> {
    CUALGO_DEVICE unsigned long long* getPointer() {
        extern CUALGO_SHMEM unsigned long long s_ulonglong[];
        return s_ulonglong;
    }
};

template <>
struct SharedMemory <bool> {
    CUALGO_DEVICE bool* getPointer() {
        extern CUALGO_SHMEM bool s_bool[];
        return s_bool;
    }
};

template <>
struct SharedMemory <float> {
    CUALGO_DEVICE float* getPointer() {
        extern CUALGO_SHMEM float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double> {
    CUALGO_DEVICE double* getPointer() {
        extern CUALGO_SHMEM double s_double[];
        return s_double;
    }
};

template <>
struct SharedMemory <uchar4> {
    CUALGO_DEVICE uchar4* getPointer() {
        extern CUALGO_SHMEM uchar4 s_uchar4[];
        return s_uchar4;
    }
};

template <>
struct SharedMemory <thrust::complex<float>> {
    CUALGO_DEVICE thrust::complex<float>* getPointer() {
        extern CUALGO_SHMEM thrust::complex<float> s_complex_float[];
        return s_complex_float;
    }
};

template <>
struct SharedMemory <thrust::complex<double>> {
    CUALGO_DEVICE thrust::complex<double>* getPointer() {
        extern CUALGO_SHMEM thrust::complex<double> s_complex_double[];
        return s_complex_double;
    }
};

#endif
