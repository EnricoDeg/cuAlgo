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

/**
 * @brief   Perform reduction on a vector
 * 
 * @details Sum elements of the input vector and return a pointer to a scalar
 * 
 * @param[in]  idata pointer to input vector
 * @param[out] odata pointer to output scalar with result of the reduction
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void reduce1dVector(int *idata, int *odata, int size);

/**
 * @brief   Perform 1D reduction on a 2D array (matrix) of size {N,K}
 * 
 * @details The reduction is done on the slow dimension, so the output
 *          vector has size {N}.
 * 
 * @param[in]  B pointer to input matrix to be reduced
 * @param[out] C pointer to output vector with result of the reduction
 * @param[in]  N contiguous dimension of the input matrix
 * @param[in]  K non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void reduce1dMatrix(int * B, int * C, size_t N, size_t K);

/**
 * @brief   Perform matrix transposition
 * 
 * @details The input matrix has dimensions {size_x, size_y} and 
 *          the output matrix has dimensions {size_y, size_x}
 * 
 * @param[in]  idata pointer to input matrix to be transposed
 * @param[out] odata pointer to output matrix with result of the transposition
 * @param[in]  size_x contiguous dimension of the input matrix
 * @param[in]  size_y non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y);

/**
 * @brief   Perform general matrix-matrix multiplication.
 * 
 * @details The following operation is performed
 *          C = alpha * A * B + beta * C
 * 
 * @param[in]    A     pointer to the input matrix.
 *                     The matrix has dimensions {K,M}.
 * @param[in]    B     pointer to the input matrix.
 *                     The matrix has dimensions {N,K}.
 * @param[inout] C     pointer to the output matrix.
 *                     The matrix has dimensions {N,M}.
 * @param[in]    M     non-contiguous dimension of the A and C matrices
 * @param[in]    N     contiguous dimension of the B and C matrix
 * @param[in]    K     contiguous dimension of the A matrix
 *                     non-contiguous dimension of the B matrix
 * @param[in]    alpha scalar parameter to apply to A * B
 * @param[in]    beta  scalar parameter to apply to C
 * 
 * @ingroup algo
 */
void gMatMul(int M, int N, int K, int alpha, const int *A,
             const int *B, int beta, int *C);

/**
 * @brief   Perform matrix-vector multiplication.
 * 
 * @details The vector A is multiplied with matrix B and the result is stored 
 *          in vector C.
 *          B * A = C
 * 
 * @param[in]  A pointer to the input vector.
 *               The vector has dimensions {K}.
 * @param[in]  B pointer to the input matrix.
 *               The matrix has dimensions {K,N}.
 * @param[out] C pointer to the output vector.
 *               The vector has dimensions {N}.
 * @param[in]  N contiguous dimension of the input matrix
 * @param[in]  K non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void gMatVecMul(int *A, int *B, int *C, size_t N, size_t K);

/**
 * @brief   Perform sparse matrix-vector multiplication with CSR 
 *          format.
 * 
 * @details The sparse matrix vector multiplication assumes that
 *          the matrix is provided in CSR format.
 *          The vector algorithm provides good performance when the
 *          number of non zero elements is high (more than 64).
 * 
 * @param[in]  columns An integer array of column positions
 *                     where the matrix value is non zero.
 * @param[in]  row_ptr Array of locations in the columns array
 *                     where a new row starts.
 * @param[in] values   An array of non zeros values of the matrix.
 * @param[in]  x       The vector array that multiplies the matrix.
 * @param[out] y       The vector array result of the multiplication.
 * @param[in]  nrows   Number of rows in the matrix.
 * 
 * @ingroup algo
 */
void gSpMatVecMulCSRVector( int * columns,
                            int * row_ptr,
                            int * values ,
                            int * x      ,
                            int * y      ,
                            int   nrows  ) ;

/**
 * @brief   Compute and return the row block array given the 
 *          row_ptr array of a matrix in CSR format.
 * 
 * @details The returned array is allocated on the device and 
 *          it can be used to call gSpMatVecMulCSRAdaptive
 * 
 * @param[in]  row_ptr      Array of locations in the columns array
 *                          where a new row starts.
 * @param[in]  nrows        Number of rows in the matrix.
 * @param[out] blocks_count Return the size of the returned array - 1
 * 
 * @return  Pointer to the device array with the number of rows
 *          for each block.
 * 
 * @ingroup algo
 */
int * getRowBlocks( const int * row_ptr     ,
                          int   nrows       ,
                          int * blocks_count);

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
 * 
 * @ingroup algo
 */
void gSpMatVecMulCSRAdaptive(int * columns     ,
                             int * row_ptr     ,
                             int * row_blocks  ,
                             int * values      ,
                             int * x           ,
                             int * y           ,
                             int   nrows       ,
                             int   blocks_count) ;

/**
 * @brief   Perform sparse matrix-vector multiplication with
 *          ELL format.
 * 
 * @details The sparse matrix vector multiplication assumes that
 *          the matrix is provided in ELL format.
 *          The ELL format is similar to the CSR but padding is
 *          used and the matrix is transposed. This means that 
 *          the first non zero elements of all the rows are 
 *          contiguous in memory in the first block.
 *          This format works well when the number of non 
 *          zero elements on each row is similar among all rows.
 *          If only a single row has a much higher number of 
 *          non zero elements, this format will significantly 
 *          increase the memory usage and the performance of
 *          the matrix vector multiplication will drop.
 * 
 * @param[in]  columns          An integer array of column positions
 *                              where the matrix value is non zero.
 * @param[in]  values           An array of non zeros values of the matrix.
 * @param[in]  x                The vector array that multiplies the matrix.
 * @param[out] y                The vector array result of the multiplication.
 * @param[in]  nrows            Number of rows in the matrix.
 * @param[in]  elements_in_rows max number of non zero elements.
 * 
 * @ingroup algo
 */
void gSpMatVecMulELL(int * columns         ,
                     int * values          ,
                     int * x               ,
                     int * y               ,
                     int   nrows           ,
                     int   elements_in_rows) ;

/**
 * @brief   Perform 1D convolution on the input matrices.
 * 
 * @details The convolution is done on the fast dimension of the input
 *          matrices R and V. This means that each convolution in the 
 *          slow dimension is independent.
 *          It can be used to convolve two groups of signals in a single kernel.
 *          The signals are expected to be in the frequency domain.
 * 
 * @param[in]  R pointer to the first input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  V pointer to the second input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[out] C pointer to the output matrix with results of the convolution.
 *               The signals are still in the frequency domain.
 *               The vector has dimension {N}.
 * @param[in]  N contiguous dimension of the input matrix
 * @param[in]  K non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void convolution1dMatrix(int * R, int * V, int * C, size_t N, size_t K);

/**
 * @brief   Perform 1D convolution on the input matrices and then a 
 *          1D reduction in the slow dimension.
 * 
 * @details This function combines convolution1dMatrix() and reduce1dMatrix()
 *          in a single kernel. The input matrices has dimensions {N,K}
 *          and the output vector has dimension {N}.
 * 
 * @param[in]  R pointer to the first input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  V pointer to the second input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[out] C pointer to the the output vector with the result of the 
 *               convolution and the reduction.
 *               The signals are still in the frequency domain already.
 *               The vector has dimension {N}.
 * @param[in]  N contiguous dimension of the input matrices
 * @param[in]  K non-contiguous dimension of the input matrices
 * 
 * @ingroup algo
 */
void convolutionReduction1dMatrix(int *  R, int *  V, int *  C, size_t N, size_t K) ;

/**
 * @brief   Perform 1D convolution on the input matrices, then apply a taper
 *          on the slow dimension and finally perform a 1D reduction in the
 *          slow dimension.
 * 
 * @details This function is similar to convolutionReduction1dMatrix() but a
 *          taper defined in the slow dimension is applied before the 1D
 *          reduction.
 * 
 * @param[in]  R pointer to the first input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  V pointer to the second input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  T pointer to the input vector with the taper values.
 *               The vector is defined in the slow dimension of the input
 *               matrices so it has dimension {K}.
 * @param[out] C pointer to the the output vector with the result of the
 *               convolution and the reduction.
 *               The signals are still in the frequency domain already.
 *               The vector has dimension {N}.
 * @param[in]  N contiguous dimension of the input matrices
 * @param[in]  K non-contiguous dimension of the input matrices
 * 
 * @ingroup algo
 */
void convolutionTaperReduction1dMatrix(int *  R, int *  V, int *  T, int *  C, size_t N, size_t K);

#endif
