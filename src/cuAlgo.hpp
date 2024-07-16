void check_cuda(cudaError_t error);

void reduce1d(int *g_idata, int *g_odata, int size);

void transposeMatrix(float *idata, float *odata, unsigned int size_x, unsigned int size_y);

void gMatMul(int M, int N, int K, float alpha, const float *A,
             const float *B, float beta, float *C);