#include <iostream>
#include <stdlib.h>
#include <cuAlgo.hpp>

int main() {

	cudaError_t err;
	unsigned int size_x = 1024;
	unsigned int size_y = 1024;
	float * input  = (float *)malloc(size_x * size_y * sizeof(float));
	float * output = (float *)malloc(size_x * size_y * sizeof(float));
	for(unsigned int j = 0; j < size_y; ++j)
		for (unsigned int i = 0; i < size_x ; ++i)
		input[i + j * size_x] = i + j * size_x;

	float *d_input;
	err = cudaMalloc(&d_input, size_x * size_y * sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	float *d_output;
	err = cudaMalloc(&d_output, size_x * size_y * sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_input, input, size_x * size_y * sizeof(float), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
	
	for (int i = 0; i < 5; ++i)
		transposeMatrix(d_input, d_output, size_x, size_y);

	err = cudaMemcpy ( output, d_output, size_x * size_y * sizeof(float), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int j = 0; j < size_y; ++j)
		for (int i = 0; i < size_x ; ++i)
			if (input[i + j * size_x] != output[j + i * size_y])
				return 1;

	return 0;
}