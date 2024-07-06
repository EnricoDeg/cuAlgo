#include <iostream>
#include <stdlib.h>
#include <cuAlgo.hpp>

int main() {

	cudaError_t err;
	int nblocks = 4096;
	int size = 1024*nblocks;
	int * input = (int *)malloc(size * sizeof(int));
	int * output = (int *)malloc(sizeof(int));
	for(int i = 0; i < nblocks; ++i)
		for (int j = 0; j < 1024 ; ++j)
		input[j + i*1024] = j;

	int *d_input;
	err = cudaMalloc(&d_input, size*sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int *d_output;
	err = cudaMalloc(&d_output, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy ( d_input, input, (size_t)size*sizeof(int), cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
	
	reduce1d(d_input, d_output, size);

	output[0] = 0;
	for(int i = 0; i < size; ++i)
		output[0] += input[i];

	std::cout << "CPU solution = " << output[0] << std::endl;

	err = cudaMemcpy ( output, d_output, sizeof(int), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		std::cout << "CUDA error (cudaMalloc): " <<  cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "GPU solution = " << output[0] << std::endl;

	return 0;
}
