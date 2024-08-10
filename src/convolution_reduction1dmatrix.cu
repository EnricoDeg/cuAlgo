/*
 * @file convolution_reduction1dmatrix.cu
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

#include <iostream>
#include "cuAlgo.hpp"
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

#define THREADS_PER_BLOCK 1024
#define COMPUTE_PER_THREAD 128

__global__ void convolutionReduction1dMatrixKernel(const int *__restrict__ R,
                                                   const int *__restrict__ V,
                                                         int *__restrict__ C,
                                                   size_t                  N,
                                                   size_t                  K,
                                                   size_t             chunks) {

	const size_t tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (tid > N / 2 * chunks)
		return;

	int tmp1 = 0;
	int tmp2 = 0;
#pragma unroll
	for (size_t i = 0; i < K / chunks; ++i) {

		size_t col = ( i * N / 2 * chunks + tid ) % ( N / 2 );
		const size_t row = ( i * N / 2 * chunks + tid ) / ( N / 2 );

		if (col == 0) {

			tmp1 += R[col + N * row] * V[col + N * row];
			tmp2 += R[col + N / 2 + N * row] * V[col + N / 2 + N * row];
		} else if (col > 0 && col < N /2) {

			tmp1 += R[col + N * row] * V[col + N * row] -
			        R[N - col + N * row] * V[N - col + N * row] ;
			tmp2 += R[N / 2 - col + N * row] * V[col + N / 2 + N * row] +
			        R[col + N / 2 + N * row] * V[N / 2 - col + N * row] ;
		}
	}

	size_t tidx = ( tid ) % ( N / 2 ) ;
	size_t tidy = ( tid ) / ( N / 2 ) ;
	C[tidx + N * tidy]         = tmp1;
	C[tidx + N / 2 + N * tidy] = tmp2;
}

void convolutionReduction1dMatrix(int *  R,
                                  int *  V,
                                  int *  C,
                                  size_t N,
                                  size_t K) {

	size_t chunks = K / COMPUTE_PER_THREAD;
	std::cout << "chunks = " << chunks << std::endl;
	std::cout << "K = " << K << std::endl;

	if (chunks > 1) {

		int * d_buffer;
		check_cuda( cudaMalloc(&d_buffer, N * chunks *sizeof(int)) );

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N / 2 * chunks, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2 * chunks, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrixKernel<<<grid, block>>>(R, V, d_buffer, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

		reduce1dMatrix(d_buffer, C, N, chunks);

		check_cuda( cudaFree ( d_buffer ) );
	} else {

		dim3 block(THREADS_PER_BLOCK);
		dim3 grid(div_ceil(N, THREADS_PER_BLOCK));

		std::cout << "threadsPerBlock = " << THREADS_PER_BLOCK << std::endl;
		std::cout << "blocksPerGrid   = " << div_ceil(N / 2, THREADS_PER_BLOCK) << std::endl;

		auto start = high_resolution_clock::now();
		convolutionReduction1dMatrixKernel<<<grid, block>>>(R, V, C, N, K, chunks);
		check_cuda( cudaDeviceSynchronize() );
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	}
}
