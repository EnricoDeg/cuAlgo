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
#include <cuda/std/complex>

#include "internals/checkError.hpp"

namespace cuAlgo{

/**
 * @brief   Perform 1D convolution with floats on the input matrices.
 * 
 * @details The convolution is done on the fast dimension of the input
 *          matrices R and V. This means that each convolution in the 
 *          slow dimension is independent.
 *          It can be used to convolve two groups of signals in a single kernel.
 *          The signals are expected to be in the frequency domain.
 *          The input matrices are expected to have the half complex memory
 *          layout. This means that this is a convolution on real signals.
 * 
 * @param[in]  R pointer to the first input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  V pointer to the second input matrix for the convolution.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[out] C pointer to the output matrix with results of the convolution.
 *               The signals are still in the frequency domain.
 *               The matrix has dimension {N,K}.
 * @param[in]  N contiguous dimension of the input matrix
 * @param[in]  K non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void convolution1dMatrixFloat(float        *R            ,
                              float        *V            ,
                              float        *C            ,
                              unsigned int  N            ,
                              unsigned int  K            ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

/**
 * @brief   Perform 1D convolution with doubles on the input matrices.
 * 
 * @details See documentation of convolution1dMatrixFloat().
 * 
 * @ingroup algo
 */
void convolution1dMatrixDouble(double       *R            ,
                               double       *V            ,
                               double       *C            ,
                               unsigned int  N            ,
                               unsigned int  K            ,
                               cudaStream_t  stream = 0   ,
                               bool          async = false);

void convolution1dMatrixInt(int          *R            ,
                            int          *V            ,
                            int          *C            ,
                            unsigned int  N            ,
                            unsigned int  K            ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Perform 1D correlation with floats on the input matrices.
 * 
 * @details The correlation is done on the fast dimension of the input
 *          matrices R and V. This means that each correlation in the 
 *          slow dimension is independent.
 *          It can be used to convolve two groups of signals in a single kernel.
 *          The signals are expected to be in the frequency domain.
 *          The input matrices are expected to have the half complex memory
 *          layout. This means that this is a correlation on real signals.
 * 
 * @param[in]  R pointer to the first input matrix for the correlation.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[in]  V pointer to the second input matrix for the correlation.
 *               The signals are assumed to be in the frequency domain already.
 *               The matrix has dimensions {N,K}.
 * @param[out] C pointer to the output matrix with results of the correlation.
 *               The signals are still in the frequency domain.
 *               The matrix has dimension {N,K}.
 * @param[in]  N contiguous dimension of the input matrix
 * @param[in]  K non-contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void correlation1dMatrixFloat(float        *R            ,
                              float        *V            ,
                              float        *C            ,
                              unsigned int  N            ,
                              unsigned int  K            ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

/**
 * @brief   Perform 1D correlation with doubles on the input matrices.
 * 
 * @details See documentation of correlation1dMatrixFloat().
 * 
 * @ingroup algo
 */
void correlation1dMatrixDouble(double       *R            ,
                               double       *V            ,
                               double       *C            ,
                               unsigned int  N            ,
                               unsigned int  K            ,
                               cudaStream_t  stream = 0   ,
                               bool          async = false);

void correlation1dMatrixInt(int          *R            ,
                            int          *V            ,
                            int          *C            ,
                            unsigned int  N            ,
                            unsigned int  K            ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Perform 2D convolution in data domain with floats on the input matrices.
 * 
 * @details The convolution is done in the data domain without Fourier
 *          Transform.
 * 
 * @param[out] odata  pointer to the output matrix for the convolution.
 *                    The matrix has dimensions {mRows+fRows-1, mCols+fCols-1}.
 * @param[in]  idata  pointer to the input matrix for the convolution.
 *                    The matrix has dimensions {mRows, mCols}.
 * @param[in]  filter pointer to the kernel of the convolution.
 *                    The matrix has dimension {fRows, fCols}.
 * @param[in]  mRows  non-contiguous dimension of the input matrix
 * @param[in]  mCols  contiguous dimension of the input matrix
 * @param[in]  fRows  non-contiguous dimensions of the kernel matrix
 * @param[in]  fCols  contiguous dimension of the kernel matrix
 * 
 * @ingroup algo
 */
void convolution2dMatrixDDFloat(float        * odata        ,
                                float        * idata        ,
                                float        * filter       ,
                                unsigned int   mRows        ,
                                unsigned int   mCols        ,
                                unsigned int   fRows        ,
                                unsigned int   fCols        ,
                                cudaStream_t   stream = 0   ,
                                bool           async = false);

/**
 * @brief   Perform 2D convolution in data domain with doubles on the input matrices.
 * 
 * @details See documentation of convolution2dMatrixDDFloat().
 * 
 * @ingroup algo
 */
void convolution2dMatrixDDDouble(double       * odata        ,
                                 double       * idata        ,
                                 double       * filter       ,
                                 unsigned int   mRows        ,
                                 unsigned int   mCols        ,
                                 unsigned int   fRows        ,
                                 unsigned int   fCols        ,
                                 cudaStream_t   stream = 0   ,
                                 bool           async = false);

/**
 * @brief   Perform 1D convolution with floats on the input matrices and then a 
 *          1D reduction in the slow dimension.
 * 
 * @details This function combines convolution1dMatrix() and reduction1dMatrix()
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
void convolutionReduction1dMatrixFloat(float        *R            ,
                                       float        *V            ,
                                       float        *C            ,
                                       unsigned int  N            ,
                                       unsigned int  K            ,
                                       cudaStream_t  stream = 0   ,
                                       bool          async = false);

/**
 * @brief   Perform 1D convolution with doubles on the input matrices and then a 
 *          1D reduction in the slow dimension.
 * 
 * @details See documentation of convolutionReduction1dMatrixFloat().
 * 
 * @ingroup algo
 */

void convolutionReduction1dMatrixDouble(double       *R            ,
                                        double       *V            ,
                                        double       *C            ,
                                        unsigned int  N            ,
                                        unsigned int  K            ,
                                        cudaStream_t  stream = 0   ,
                                        bool          async = false);

void convolutionReduction1dMatrixInt(int          *R            ,
                                     int          *V            ,
                                     int          *C            ,
                                     unsigned int  N            ,
                                     unsigned int  K            ,
                                     cudaStream_t  stream = 0   ,
                                     bool          async = false) ;

/**
 * @brief   Perform 1D convolution with floats on the input matrices, then apply a taper
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
void convolutionTaperReduction1dMatrixFloat(float        *R            ,
                                            float        *V            ,
                                            float        *T            ,
                                            float        *C            ,
                                            unsigned int  N            ,
                                            unsigned int  K            ,
                                            cudaStream_t  stream = 0   ,
                                            bool          async = false);

/**
 * @brief   Perform 1D convolution with doubles on the input matrices, then apply a taper
 *          on the slow dimension and finally perform a 1D reduction in the
 *          slow dimension.
 * 
 * @details See documentation of convolutionTaperReduction1dMatrixFloat().
 * 
 * @ingroup algo
 */
void convolutionTaperReduction1dMatrixDouble(double       *R            ,
                                             double       *V            ,
                                             double       *T            ,
                                             double       *C            ,
                                             unsigned int  N            ,
                                             unsigned int  K            ,
                                             cudaStream_t  stream = 0   ,
                                             bool          async = false);

void convolutionTaperReduction1dMatrixInt(int          *R            ,
                                          int          *V            ,
                                          int          *T            ,
                                          int          *C            ,
                                          unsigned int  N            ,
                                          unsigned int  K            ,
                                          cudaStream_t  stream = 0   ,
                                          bool          async = false);

/**
 * @brief   Compute dot product of two vectors of floats
 * 
 * @details Sum product of elements of the input vectors and
 *          return a pointer to a scalar
 * 
 * @param[in]  g_idata1 pointer to input vector
 * @param[in]  g_idata2 pointer to input vector
 * @param[out] g_odata pointer to output scalar with result of the L1 norm
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void dotProduct1dVectorFloat(float        *g_idata1     ,
                             float        *g_idata2     ,
                             float        *g_odata      ,
                             unsigned int  size         ,
                             cudaStream_t  stream = 0   ,
                             bool          async = false);

/**
 * @brief   Compute dot product of two vectors of double
 * 
 * @details See documentation of dotProductVectorFloat().
 * 
 * @ingroup algo
 */
void dotProduct1dVectorDouble(double       *g_idata1     ,
                              double       *g_idata2     ,
                              double       *g_odata      ,
                              unsigned int  size         ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

void dotProduct1dVectorInt(int          *g_idata1     ,
                           int          *g_idata2     ,
                           int          *g_odata      ,
                           unsigned int  size         ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   downsample operator in 1d on matrix of floats
 * 
 * @details The operation can be applied on both direction
 * 
 * @param[in]  idata pointer to input matrix
 * @param[out] odata pointer to output matrix
 * @param[in]  dim dimension where to apply the dshear operation.
 *             0 for rows and 1 for columns.
 * @param[in]  stride downsample stride on dimension `dim`
 * @param[in]  mRows non-contiguous dimension of the input matrix
 * @param[in]  mCols contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void downsample1dMatrixFloat(float        *idata        ,
                             float        *odata        ,
                             unsigned int  dim          ,
                             unsigned int  stride       ,
                             unsigned int  mRows        ,
                             unsigned int  mCols        ,
                             cudaStream_t  stream = 0   ,
                             bool          async = false);

/**
 * @brief   downsample operator in 1d on matrix of doubles
 * 
 * @details See documentation of downsample1dMatrixFloat().
 * 
 * @ingroup algo
 */
void downsample1dMatrixDouble(double       *idata        ,
                              double       *odata        ,
                              unsigned int  dim          ,
                              unsigned int  stride       ,
                              unsigned int  mRows        ,
                              unsigned int  mCols        ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

/**
 * @brief   dshear operator in 1d on matrix of floats
 * 
 * @details The operation can be applied on both direction
 * 
 * @param[in]  idata pointer to input matrix
 * @param[out] odata pointer to output matrix
 * @param[in]  k  long int to define the shift
 * @param[in]  dim dimension where to apply the dshear operation.
 *             0 for rows and 1 for columns.
 * @param[in]  mRows non-contiguous dimension of the input and output matrices
 * @param[in]  mCols contiguous dimension of the input and output matrices
 * 
 * @ingroup algo
 */
void dshear1dMatrixFloat(float        *idata        ,
                         float        *odata        ,
                         long int      k            ,
                         unsigned int  dim          ,
                         unsigned int  mRows        ,
                         unsigned int  mCols        ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

/**
 * @brief   dshear operator in 1d on matrix of doubles
 * 
 * @details See documentation of dshear1dMatrixDouble().
 * 
 * @ingroup algo
 */
void dshear1dMatrixDouble(double       *idata        ,
                          double       *odata        ,
                          long int      k            ,
                          unsigned int  dim          ,
                          unsigned int  mRows        ,
                          unsigned int  mCols        ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

/**
 * @brief   Perform exclusive scan or prefix sum on a vector using
 *          floats.
 * 
 * @details The input and output arrays are expected to be multiple of 
 *          1024. If not, they should be padded before calling the 
 *          function.
 * 
 * @param[in]  g_idata input array of size {size}
 * @param[out] g_odata output array of size {size}
 * @param[in]  size size of input and output arrays
 * @param[in]  stream CUDA stream were the kernel is launched.
 *                    Default is 0.
 * @param[in]  async boolean defining if the kernel should be
 *                   asynchronous. Default is false.
 * 
 * @ingroup algo
 */
void exclusiveScan1dVectorFloat(float        *g_idata      ,
                                float        *g_odata      ,
                                unsigned int  size         ,
                                cudaStream_t  stream = 0   ,
                                bool          async = false);

/**
 * @brief   Perform exclusive scan or prefix sum on a vector using
 *          doubles.
 * 
 * @details See documentation of exclusiveScan1dVectorFloat().
 * 
 * @ingroup algo
 */
void exclusiveScan1dVectorDouble(double       *g_idata      ,
                                 double       *g_odata      ,
                                 unsigned int  size         ,
                                 cudaStream_t  stream = 0   ,
                                 bool          async = false);

void exclusiveScan1dVectorInt(int          *g_idata      ,
                              int          *g_odata      ,
                              unsigned int  size         ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

/**
 * @brief   Perform fftshift on a vector with floats
 * 
 * @details The input vector has dimension {size} and 
 *          the output vector has dimension {size}
 * 
 * @param[in]  idata pointer to input vector to be shifted
 * @param[out] odata pointer to output vector with result of the fftshift
 * @param[in]  size  contiguous dimension of the input and output vectors
 * 
 * @ingroup algo
 */
void fftshift1dVectorFloat(float        *idata        ,
                           float        *odata        ,
                           unsigned int  size         ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   Perform fftshift on a vector with doubles
 * 
 * @details See documentation of fftshift1dVectorFloat()
 * 
 * @ingroup algo
 */
void fftshift1dVectorDouble(double       *idata        ,
                            double       *odata        ,
                            unsigned int  size         ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

void fftshift1dVectorInt(int          *idata        ,
                         int          *odata        ,
                         unsigned int  size         ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

/**
 * @brief   Perform fftshift on a matrix with floats in place
 * 
 * @details The fftshift operation is performed on both
 *          dimensions
 * 
 * @param[inout] data  pointer to matrix to be shifted
 * @param[in]    mRows non-contiguous dimension of the matrix
 * @param[in]    mCols contiguous dimension of the matrix
 * 
 * @ingroup algo
 */
void fftshift2dMatrixFloat(float        *data         ,
                           unsigned int  mRows        ,
                           unsigned int  mCols        ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   Perform fftshift on a matrix with doubles in place
 * 
 * @details See documentation of fftshift2dMatrixFloat()
 * 
 * @ingroup algo
 */
void fftshift2dMatrixDouble(double       *data         ,
                            unsigned int  mRows        ,
                            unsigned int  mCols        ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Flip rows or columns of a matrix of floats in place
 * 
 * @details The `dim` argument defines if rows or columns 
 *          should be flipped.
 * 
 * @param[inout] data  pointer to matrix to be flipped
 * @param[in]    dim   dimension to flip
 * @param[in]    mRows non-contiguous dimension of the matrix
 * @param[in]    mCols contiguous dimension of the matrix
 * 
 * @ingroup algo
 */
void fliplr1dMatrixFloat(float        *data         ,
                         unsigned int  dim          ,
                         unsigned int  mRows        ,
                         unsigned int  mCols        ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

/**
 * @brief   Flip rows or columns of a matrix of doubles in place
 * 
 * @details See documentation of fliplr1dMatrixFloat()
 * 
 * @ingroup algo
 */
void fliplr1dMatrixDouble(double       *data         ,
                          unsigned int  dim          ,
                          unsigned int  mRows        ,
                          unsigned int  mCols        ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

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
unsigned int * getRowBlocks( const unsigned int * row_ptr     ,
                                   unsigned int   nrows       ,
                                   unsigned int * blocks_count);

/**
 * @brief   Perform general matrix-matrix multiplication with floats.
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
void gMatMulFloat(float         alpha        ,
                  const float  *A            ,
                  const float  *B            ,
                  float         beta         ,
                  float        *C            ,
                  unsigned int  M            ,
                  unsigned int  N            ,
                  unsigned int  K            ,
                  cudaStream_t  stream = 0   ,
                  bool          async = false);

/**
 * @brief   Perform general matrix-matrix multiplication with doubles.
 * 
 * @details See documentation of gMatMulFloat().
 * 
 * @ingroup algo
 */
void gMatMulDouble(double        alpha        ,
                   const double *A            ,
                   const double *B            ,
                   double        beta         ,
                   double       *C            ,
                   unsigned int  M            ,
                   unsigned int  N            ,
                   unsigned int  K            ,
                   cudaStream_t  stream = 0   ,
                   bool          async = false);

void gMatMulInt(int           alpha        ,
                const int    *A            ,
                const int    *B            ,
                int           beta         ,
                int          *C            ,
                unsigned int  M            ,
                unsigned int  N            ,
                unsigned int  K            ,
                cudaStream_t  stream = 0   ,
                bool          async = false);

/**
 * @brief   Perform matrix-vector multiplication with floats.
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
void gMatVecMulFloat(const float        *A            ,
                     const float        *B            ,
                           float        *C            ,
                           unsigned int  N            ,
                           unsigned int  K            ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   Perform matrix-vector multiplication with doubles.
 * 
 * @details See documentation of gMatVecMulFloat().
 * 
 * @ingroup algo
 */
void gMatVecMulDouble(const double       *A            ,
                      const double       *B            ,
                            double       *C            ,
                            unsigned int  N            ,
                            unsigned int  K            ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

void gMatVecMulInt(const int          *A            ,
                   const int          *B            ,
                         int          *C            ,
                         unsigned int  N            ,
                         unsigned int  K            ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

/**
 * @brief   Compute the matrix gradient using floats.
 * 
 * @details The gradient is computed using first order
 *          accuracy.
 * 
 * @param[in]  A input matrix of size {N,M}
 * @param[out] Ax output matrix with x derivative.
 *             The matrix has the same size of A.
 * @param[out] Ay output matrix with y derivative.
 *             The matrix has the same size of A.
 * @param[in]  M size of contiguous dimension
 * @param[in]  N size of non-contiguous dimension
 * @param[in]  stream CUDA stream were the kernel is launched.
 *                    Default is 0.
 * @param[in]  async boolean defining if the kernel should be
 *                   asynchronous. Default is false.
 * 
 * @ingroup algo
 */
void grad2dMatrixFloat(float        *A            ,
                       float        *Ax           ,
                       float        *Ay           ,
                       unsigned int  M            ,
                       unsigned int  N            ,
                       cudaStream_t  stream = 0   ,
                       bool          async = false);

/**
 * @brief   Compute the matrix gradient using doubles.
 * 
 * @details See documentation of gradMatrixFloat().
 * 
 * @ingroup algo
 */
void grad2dMatrixDouble(double       *A            ,
                        double       *Ax           ,
                        double       *Ay           ,
                        unsigned int  M            ,
                        unsigned int  N            ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

void grad2dMatrixInt(int          *A            ,
                     int          *Ax           ,
                     int          *Ay           ,
                     unsigned int  M            ,
                     unsigned int  N            ,
                     cudaStream_t  stream = 0   ,
                     bool          async = false);

/**
 * @brief   Perform sparse matrix-vector multiplication with an 
 *          adaptive methodn using floats.
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
void gSpMatVecMulCSRAdaptiveFloat(unsigned int   *columns      ,
                                  unsigned int   *row_ptr      ,
                                  unsigned int   *row_blocks   ,
                                           float *values       ,
                                           float *x            ,
                                           float *y            ,
                                  unsigned int    nrows        ,
                                  unsigned int    blocks_count ,
                                  cudaStream_t    stream = 0   ,
                                  bool            async = false);

/**
 * @brief   Perform sparse matrix-vector multiplication with an 
 *          adaptive methodn using doubles.
 * 
 * @details See documentation of gSpMatVecMulCSRAdaptiveFloat().
 * 
 * @ingroup algo
 */
void gSpMatVecMulCSRAdaptiveDouble(unsigned int    *columns      ,
                                   unsigned int    *row_ptr      ,
                                   unsigned int    *row_blocks   ,
                                            double *values       ,
                                            double *x            ,
                                            double *y            ,
                                   unsigned int     nrows        ,
                                   unsigned int     blocks_count ,
                                   cudaStream_t     stream = 0   ,
                                   bool             async = false);

void gSpMatVecMulCSRAdaptiveInt(unsigned int *columns      ,
                                unsigned int *row_ptr      ,
                                unsigned int *row_blocks   ,
                                         int *values       ,
                                         int *x            ,
                                         int *y            ,
                                unsigned int  nrows        ,
                                unsigned int  blocks_count ,
                                cudaStream_t  stream = 0   ,
                                bool          async = false);

/**
 * @brief   Perform sparse matrix-vector multiplication with CSR 
 *          format using floats.
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
void gSpMatVecMulCSRVectorFloat(unsigned int *columns      ,
                                unsigned int *row_ptr      ,
                                float        *values       ,
                                float        *x            ,
                                float        *y            ,
                                unsigned int  nrows        ,
                                cudaStream_t  stream = 0   ,
                                bool          async = false);

/**
 * @brief   Perform sparse matrix-vector multiplication with CSR 
 *          format using doubles.
 * 
 * @details See documentation of gSpMatVecMulCSRVectorFloat().
 * 
 * @ingroup algo
 */
void gSpMatVecMulCSRVectorDouble(unsigned int *columns      ,
                                 unsigned int *row_ptr      ,
                                 double       *values       ,
                                 double       *x            ,
                                 double       *y            ,
                                 unsigned int  nrows        ,
                                 cudaStream_t  stream = 0   ,
                                 bool          async = false);

void gSpMatVecMulCSRVectorInt(unsigned int *columns      ,
                              unsigned int *row_ptr      ,
                              int          *values       ,
                              int          *x            ,
                              int          *y            ,
                              unsigned int  nrows        ,
                              cudaStream_t  stream = 0   ,
                              bool          async = false);

/**
 * @brief   Perform sparse matrix-vector multiplication with
 *          ELL format using floats.
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
void gSpMatVecMulELLFloat(unsigned int *columns         ,
                          float        *values          ,
                          float        *x               ,
                          float        *y               ,
                          unsigned int  nrows           ,
                          unsigned int  elements_in_rows,
                          cudaStream_t  stream = 0      ,
                          bool          async = false   );

/**
 * @brief   Perform sparse matrix-vector multiplication with
 *          ELL format using doubles.
 * 
 * @details See documentation of gSpMatVecMulELLFloat().
 * 
 * @ingroup algo
 */
void gSpMatVecMulELLDouble(unsigned int *columns         ,
                           double       *values          ,
                           double       *x               ,
                           double       *y               ,
                           unsigned int  nrows           ,
                           unsigned int  elements_in_rows,
                           cudaStream_t  stream = 0      ,
                           bool          async = false   );

void gSpMatVecMulELLInt(unsigned int *columns         ,
                        int          *values          ,
                        int          *x               ,
                        int          *y               ,
                        unsigned int  nrows           ,
                        unsigned int  elements_in_rows,
                        cudaStream_t  stream = 0      ,
                        bool          async = false   );

/**
 * @brief   Perform ifftshift on a matrix with floats in place
 * 
 * @details The ifftshift operation is performed on both
 *          dimensions
 * 
 * @param[inout] data  pointer to matrix to be shifted
 * @param[in]    mRows non-contiguous dimension of the matrix
 * @param[in]    mCols contiguous dimension of the matrix
 * 
 * @ingroup algo
 */
void ifftshift2dMatrixFloat(float        *data         ,
                            unsigned int  mRows        ,
                            unsigned int  mCols        ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Perform ifftshift on a matrix with doubles in place
 * 
 * @details See documentation of ifftshift2dMatrixFloat().
 * 
 * @ingroup algo
 */
void ifftshift2dMatrixDouble(double       *data         ,
                             unsigned int  mRows        ,
                             unsigned int  mCols        ,
                             cudaStream_t  stream = 0   ,
                             bool          async = false);

/**
 * @brief   Normalize a vector of floats
 * 
 * @details Each element of the vector is divided by the L1 norm
 *          of the vector
 * 
 * @param[inout]  idata pointer to input vector which will be normalized
 * @param[in]     size  size of the input vector
 * 
 * @ingroup algo
 */
void normalizeVectorFloat(float        *g_idata      ,
                          unsigned int  size         ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

/**
 * @brief   Normalize a vector of doubles
 * 
 * @details See documentation of normalizeVectorFloat().
 * 
 * @ingroup algo
 */
void normalizeVectorDouble(double       *g_idata      ,
                           unsigned int  size         ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

void normalizeVectorInt(int          *g_idata      ,
                        unsigned int  size         ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

/**
 * @brief   Compute L1 norm on a vector of floats
 * 
 * @details Sum absolute value of elements of the input vector and
 *          return a pointer to a scalar
 * 
 * @param[in]  idata pointer to input vector
 * @param[out] odata pointer to output scalar with result of the L1 norm
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void normL1VectorFloat(float        *g_idata      ,
                       float        *g_odata      ,
                       unsigned int  size         ,
                       cudaStream_t  stream = 0   ,
                       bool          async = false);

/**
 * @brief   Compute L1 norm on a vector of doubles
 * 
 * @details See documentation of normL1VectorFloat().
 * 
 * @ingroup algo
 */
void normL1VectorDouble(double       *g_idata      ,
                        double       *g_odata      ,
                        unsigned int  size         ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

void normL1VectorInt(int          *g_idata      ,
                     int          *g_odata      ,
                     unsigned int  size         ,
                     cudaStream_t  stream = 0   ,
                     bool          async = false);

/**
 * @brief   Compute squared L2 norm on a vector of floats
 * 
 * @details Sum squared value of elements of the input vector and
 *          return a pointer to a scalar
 * 
 * @param[in]  idata pointer to input vector
 * @param[out] odata pointer to output scalar with result of the squared L2 norm
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void normL2VectorFloat(float        *g_idata      ,
                       float        *g_odata      ,
                       unsigned int  size         ,
                       cudaStream_t  stream = 0   ,
                       bool          async = false);

/**
 * @brief   Compute L2 norm on a vector of doubles
 * 
 * @details See documentation of normL2VectorFloat().
 * 
 * @ingroup algo
 */
void normL2VectorDouble(double       *g_idata      ,
                        double       *g_odata      ,
                        unsigned int  size         ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

void normL2VectorInt(int          *g_idata      ,
                     int          *g_odata      ,
                     unsigned int  size         ,
                     cudaStream_t  stream = 0   ,
                     bool          async = false);

/**
 * @brief   Compute LInfinity norm on a vector of floats
 * 
 * @details Max of absolute value of elements of the input vector and
 *          return a pointer to a scalar
 * 
 * @param[in]  idata pointer to input vector
 * @param[out] odata pointer to output scalar with result of the L infinity norm
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void normLInfVectorFloat(float        *g_idata      ,
                         float        *g_odata      ,
                         unsigned int  size         ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

/**
 * @brief   Compute LInfinity norm on a vector of doubles
 * 
 * @details See documentation of normLInfVectorFloat().
 * 
 * @ingroup algo
 */
void normLInfVectorDouble(double       *g_idata      ,
                          double       *g_odata      ,
                          unsigned int  size         ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

void normLInfVectorInt(int          *g_idata      ,
                       int          *g_odata      ,
                       unsigned int  size         ,
                       cudaStream_t  stream = 0   ,
                       bool          async = false);

/**
 * @brief   Pad matrix of floats
 * 
 * @param[in]  idata pointer to input matrix
 * @param[out] odata pointer to output matrix
 * @param[in]  nRows non-contiguous dimension of odata
 * @param[in]  nCols contiguous dimension of odata
 * @param[in]  mRows non-contiguous dimension of idata
 * @param[in]  mCols contiguous dimension of idata
 * 
 * @ingroup algo
 */
void padarray2dMatrixFloat(float        *idata        ,
                           float        *odata        ,
                           unsigned int  nRows        ,
                           unsigned int  nCols        ,
                           unsigned int  mRows        ,
                           unsigned int  mCols        ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   Pad matrix of doubles
 * 
 * @details See documentation of padarray2dMatrixFloat
 * 
 * @ingroup algo
 */
void padarray2dMatrixDouble(double       *idata        ,
                            double       *odata        ,
                            unsigned int  nRows        ,
                            unsigned int  nCols        ,
                            unsigned int  mRows        ,
                            unsigned int  mCols        ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Pad matrix of complex floats
 * 
 * @details See documentation of padarray2dMatrixFloat
 * 
 * @ingroup algo
 */
void padarray2dMatrixComplexFloat(cuda::std::complex<float> *idata        ,
                                  cuda::std::complex<float> *odata        ,
                                  unsigned int               nRows        ,
                                  unsigned int               nCols        ,
                                  unsigned int               mRows        ,
                                  unsigned int               mCols        ,
                                  cudaStream_t               stream = 0   ,
                                  bool                       async = false);

/**
 * @brief   Pad matrix of complex doubles
 * 
 * @details See documentation of padarray2dMatrixFloat
 * 
 * @ingroup algo
 */
void padarray2dMatrixComplexDouble(cuda::std::complex<double> *idata        ,
                                   cuda::std::complex<double> *odata        ,
                                   unsigned int                nRows        ,
                                   unsigned int                nCols        ,
                                   unsigned int                mRows        ,
                                   unsigned int                mCols        ,
                                   cudaStream_t                stream = 0   ,
                                   bool                        async = false);

/**
 * @brief   Perform 1D reduction with floats on a 2D array (matrix)
 *          of size {N,K}
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
void reduction1dMatrixFloat(float        *B            ,
                            float        *C            ,
                            unsigned int  N            ,
                            unsigned int  K            ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Perform 1D reduction with doubles on a 2D array (matrix)
 *          of size {N,K}
 * 
 * @details See documentation of reduction1dMatrixFloat()
 * 
 * @ingroup algo
 */

void reduction1dMatrixDouble(double       *B            ,
                             double       *C            ,
                             unsigned int  N            ,
                             unsigned int  K            ,
                             cudaStream_t  stream = 0   ,
                             bool          async = false);

void reduction1dMatrixInt(int          *B            ,
                          int          *C            ,
                          unsigned int  N            ,
                          unsigned int  K            ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

/**
 * @brief   Perform reduction on a vector of floats
 * 
 * @details Sum elements of the input vector and return a pointer to a scalar
 * 
 * @param[in]  idata pointer to input vector
 * @param[out] odata pointer to output scalar with result of the reduction
 * @param[in]  size  size of the input vector
 * 
 * @ingroup algo
 */
void reduction1dVectorFloat(float        *g_idata      ,
                            float        *g_odata      ,
                            unsigned int  size         ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

/**
 * @brief   Perform reduction on a vector of doubles
 * 
 * @details See documentation of reduction1dVectorFloat().
 * 
 * @ingroup algo
 */
void reduction1dVectorDouble(double       *g_idata      ,
                             double       *g_odata      ,
                             unsigned int  size         ,
                             cudaStream_t  stream = 0   ,
                             bool          async = false);

void reduction1dVectorInt(int          *g_idata      ,
                          int          *g_odata      ,
                          unsigned int  size         ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

/**
 * @brief   Apply 1d taper to matrix using floats.
 * 
 * @details The taper is applied on the fastest 
 *          dimension. Each row of the matrix is associated
 *          with a startIndex and an endIndex. The values
 *          less than startIndex are set to 0. The values
 *          greater than endIndex are also set to 0.
 * 
 * @param[inout]  A            input matrix of size {N,M}
 * @param[in]     taper        taper array of sioze taperLength
 * @param[in]     startIndices start indices to apply taper. 
 *                Values with index less than startIndices are
 *                set to 0. 
 *                The array is of size {N}.
 * @param[in]     endIndices   end indices to apply taper. 
 *                Values with index greater than endIndices are
 *                set to 0. 
 *                The array is of size {N}.
 * @param[in]     M            size of contiguous dimension.
 * @param[in]     N            size of non-contiguous dimension.
 * @param[in]     taperLength  size of taper array.
 * @param[in]     stream       CUDA stream were the kernel is launched.
 *                             Default is 0.
 * @param[in]     async        boolean defining if the kernel should be
 *                             asynchronous. Default is false.
 * 
 * @ingroup algo
 */
void taper1dMatrixFloat(float        *A            ,
                        float        *taper        ,
                        unsigned int *startIndices ,
                        unsigned int *endIndices   ,
                        unsigned int  M            ,
                        unsigned int  N            ,
                        unsigned int  taperLength  ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

/**
 * @brief   Apply 1d taper to matrix using doubles.
 * 
 * @details See documentation of taper1dMatrixFloat().
 * 
 * @ingroup algo
 */
void taper1dMatrixDouble(double       *A            ,
                         double       *taper        ,
                         unsigned int *startIndices ,
                         unsigned int *endIndices   ,
                         unsigned int  M            ,
                         unsigned int  N            ,
                         unsigned int  taperLength  ,
                         cudaStream_t  stream = 0   ,
                         bool          async = false);

void taper1dMatrixInt(int          *A            ,
                      int          *taper        ,
                      unsigned int *startIndices ,
                      unsigned int *endIndices   ,
                      unsigned int  M            ,
                      unsigned int  N            ,
                      unsigned int  taperLength  ,
                      cudaStream_t  stream = 0   ,
                      bool          async = false);

/**
 * @brief   Perform matrix transposition with floats
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
void transposeMatrixFloat(float        *idata        ,
                          float        *odata        ,
                          unsigned int  size_x       ,
                          unsigned int  size_y       ,
                          cudaStream_t  stream = 0   ,
                          bool          async = false);

/**
 * @brief   Perform matrix transposition with doubles
 * 
 * @details See documentation of transposeMatrixFloat()
 * 
 * @ingroup algo
 */
void transposeMatrixDouble(double       *idata        ,
                           double       *odata        ,
                           unsigned int  size_x       ,
                           unsigned int  size_y       ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

void transposeMatrixInt(int          *idata        ,
                        int          *odata        ,
                        unsigned int  size_x       ,
                        unsigned int  size_y       ,
                        cudaStream_t  stream = 0   ,
                        bool          async = false);

/**
 * @brief   upsample operator in 1d on matrix of floats
 * 
 * @details The operation can be applied on both direction
 * 
 * @param[in]  idata  pointer to input matrix
 * @param[out] odata  pointer to output matrix
 * @param[in]  dim    dimension where to apply the dshear operation.
 *                    0 for rows and 1 for columns.
 * @param[in]  nzeros number of zeros on dimension `dim` before copying 
 *                    one element of the input matrix
 * @param[in]  mRows  non-contiguous dimension of the input matrix
 * @param[in]  mCols  contiguous dimension of the input matrix
 * 
 * @ingroup algo
 */
void upsample1dMatrixFloat(float        *idata        ,
                           float        *odata        ,
                           unsigned int  dim          ,
                           unsigned int  nzeros       ,
                           unsigned int  mRows        ,
                           unsigned int  mCols        ,
                           cudaStream_t  stream = 0   ,
                           bool          async = false);

/**
 * @brief   upsample operator in 1d on matrix of doubles
 * 
 * @details See documentation of upsample1dMatrixFloat()
 * 
 * @ingroup algo
 */
void upsample1dMatrixDouble(double       *idata        ,
                            double       *odata        ,
                            unsigned int  dim          ,
                            unsigned int  nzeros       ,
                            unsigned int  mRows        ,
                            unsigned int  mCols        ,
                            cudaStream_t  stream = 0   ,
                            bool          async = false);

}
#endif
