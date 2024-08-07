#ifndef CUALGO_H
#define CUALGO_H

#include <cuda.h>

void check_cuda(cudaError_t error);

void reduce1dVector(int *g_idata, int *g_odata, int size);

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y);

void gMatMul(int M, int N, int K, float alpha, const float *A,
             const float *B, float beta, float *C);

void reduce1dMatrix(int * B, int * C, size_t N, size_t K);

void convolution1dMatrix(int * R, int * V, int * C, size_t N, size_t K);

#endif
