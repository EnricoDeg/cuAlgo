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

#include "cuAlgo/internals/utils.hpp"

template <typename T>
class reductionSum_impl {

    public:
    CUALGO_DEVICE inline T globalMemory(T * CUALGO_RESTRICT a,
                                        T * CUALGO_RESTRICT b ) {
        return *a + *b;
    }
    CUALGO_DEVICE inline void loadSharedMemory(T * result,
                                               T * CUALGO_RESTRICT a) {
        *result += *a;
    }
    CUALGO_DEVICE inline void sharedMemory(T * result, T * data) {
        *result += *data;
    }
    CUALGO_DEVICE inline void atomic(T * address, T value) {
        atomicAdd(address, value);
    }
};

template <typename T>
class normL1_impl {

    public:
    CUALGO_DEVICE inline T globalMemory(T * CUALGO_RESTRICT a,
                                        T * CUALGO_RESTRICT b ) {
        return abs(*a) + abs(*b);
    }
    CUALGO_DEVICE inline void loadSharedMemory(T * result,
                                               T * CUALGO_RESTRICT a) {
        *result += *a;
    }
    CUALGO_DEVICE inline void sharedMemory(T * result, T * data) {
        *result += *data;
    }
    CUALGO_DEVICE inline void atomic(T * address, T value) {
        atomicAdd(address, value);
    }
};

template <typename T>
class normL2_impl {

    public:
    CUALGO_DEVICE inline T globalMemory(T * CUALGO_RESTRICT a,
                                        T * CUALGO_RESTRICT b ) {
        return 	abs(*a) * abs(*a) + abs(*b) * abs(*b);
    }
    CUALGO_DEVICE inline void loadSharedMemory(T * result,
                                               T * CUALGO_RESTRICT a) {
        *result += *a;
    }
    CUALGO_DEVICE inline void sharedMemory(T * result, T * data) {
        *result += *data;
    }
    CUALGO_DEVICE inline void atomic(T * address, T value) {
        atomicAdd(address, value);
    }
};

template <typename T>
class normLInf_impl {

    public:
    CUALGO_DEVICE inline T globalMemory(T * CUALGO_RESTRICT a,
                                        T * CUALGO_RESTRICT b ) {
        return max(abs(*a), abs(*b));
    }
    CUALGO_DEVICE inline void loadSharedMemory(T * result,
                                               T * CUALGO_RESTRICT a) {
        *result = max(*result, *a);
    }
    CUALGO_DEVICE inline void sharedMemory(T * result, T * data) {
        *result = max(*result, *data);
    }
    CUALGO_DEVICE inline void atomic(T * addr, T val) {
        if (*addr >= val) return;

        unsigned int *const addr_as_ui = (unsigned int *)addr;
        unsigned int old = *addr_as_ui, assumed;
        do {
            assumed = old;
            if (__uint_as_float(assumed) >= val) break;
            old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
        } while (assumed != old);
    }
};

template <typename T>
class dotProduct_impl {

    public:
    CUALGO_DEVICE inline T globalMemory(T * CUALGO_RESTRICT init,
                                        T * CUALGO_RESTRICT a1,
                                        T * CUALGO_RESTRICT a2) {
        return (*init) + (*a1) * (*a2);
    }
    CUALGO_DEVICE inline void loadSharedMemory(T * result,
                                               T * CUALGO_RESTRICT a) {
        *result += *a;
    }
    CUALGO_DEVICE inline void sharedMemory(T * result, T * data) {
        *result += *data;
    }
    CUALGO_DEVICE inline void atomic(T * address, T value) {
        atomicAdd(address, value);
    }
};

template<typename T>
class convolution_impl {

    public:
    CUALGO_DEVICE inline T firstRealImag(const T * CUALGO_RESTRICT a,
                                         const T * CUALGO_RESTRICT b) {
        return (*a) * (*b);
    }
    CUALGO_DEVICE inline T nReal(const T * CUALGO_RESTRICT a,
                                 const T * CUALGO_RESTRICT b,
                                 const T * CUALGO_RESTRICT c,
                                 const T * CUALGO_RESTRICT d) {
        return (*a) * (*b) - (*c) * (*d);
    }
    CUALGO_DEVICE inline T nImag(const T * CUALGO_RESTRICT a,
                                 const T * CUALGO_RESTRICT b,
                                 const T * CUALGO_RESTRICT c,
                                 const T * CUALGO_RESTRICT d) {
        return (*a) * (*b) + (*c) * (*d);
    }
};

template<typename T>
class correlation_impl {

    public:
    CUALGO_DEVICE inline T firstRealImag(const T * CUALGO_RESTRICT a,
                                         const T * CUALGO_RESTRICT b) {
        return (*a) * (*b);
    }
    CUALGO_DEVICE inline T nReal(const T * CUALGO_RESTRICT a,
                                 const T * CUALGO_RESTRICT b,
                                 const T * CUALGO_RESTRICT c,
                                 const T * CUALGO_RESTRICT d) {
        return (*a) * (*b) + (*c) * (*d);
    }
    CUALGO_DEVICE inline T nImag(const T * CUALGO_RESTRICT a,
                                 const T * CUALGO_RESTRICT b,
                                 const T * CUALGO_RESTRICT c,
                                 const T * CUALGO_RESTRICT d) {
        return (*a) * (*b) - (*c) * (*d);
    }
};

#endif
