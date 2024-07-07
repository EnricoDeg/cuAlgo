#include <iostream>
#include <cuda.h>
#include "cuAlgo.hpp"
#include <chrono>

void check_cuda(cudaError_t error) {

	if ( error != cudaSuccess ) {
		std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
		exit(EXIT_FAILURE);
	}
}