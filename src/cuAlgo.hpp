/*
 * @file cuAlgo.hpp
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
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

/**
 * @brief   Check CUDA error
 * 
 * @details Check error returned by CUDA function and print error string
 *          if it is not cudaSuccess
 * 
 * @param[in] error cuda error
 * 
 * @ingroup algo
 */
void check_cuda(cudaError_t error);

void reduce1dVector(int *g_idata, int *g_odata, int size);

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y);

void gMatMul(int M, int N, int K, int alpha, const int *A,
             const int *B, int beta, int *C);

void reduce1dMatrix(int * B, int * C, size_t N, size_t K);

void convolution1dMatrix(int * R, int * V, int * C, size_t N, size_t K);

void convolutionReduction1dMatrix(int *  R, int *  V, int *  C, size_t N, size_t K) ;

void convolutionTaperReduction1dMatrix(int *  R, int *  V, int *  T, int *  C, size_t N, size_t K);

#endif
