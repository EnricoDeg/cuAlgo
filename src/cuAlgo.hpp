/*
 * @file cuAlgo.hpp
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
#ifndef CUALGO_H
#define CUALGO_H

#include <cuda.h>

#include "checkError.hpp"

void reduce1dVector(int *g_idata, int *g_odata, int size);

void reduce1dMatrix(int * B, int * C, size_t N, size_t K);

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y);

void gMatMul(int M, int N, int K, int alpha, const int *A,
             const int *B, int beta, int *C);

void gMatVecMul(int *A, int *B, int *C, size_t N, size_t K);

void convolution1dMatrix(int * R, int * V, int * C, size_t N, size_t K);

void convolutionReduction1dMatrix(int *  R, int *  V, int *  C, size_t N, size_t K) ;

void convolutionTaperReduction1dMatrix(int *  R, int *  V, int *  T, int *  C, size_t N, size_t K);

#endif
