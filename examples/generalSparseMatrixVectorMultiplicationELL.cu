/*
 * @file generalSparseMatrixVectorMultiplicationELL.cu
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

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <ctime> 
#include <cuAlgo.hpp>

int main(int argc, char *argv[])
{

	srand((unsigned)time(0)); 

	const unsigned int nrows = 1024;
	const unsigned int nnz   = 32 ;

	// Allocate enough storage for the matix.  We allocate more than
	// is needed in order to simplify the code
	unsigned int * columns  = (unsigned int *)malloc(   nrows * nnz  * sizeof(unsigned int));
	int          * values   = (         int *)malloc(   nrows * nnz  * sizeof(         int));
	int          * x        = (         int *)malloc(   nrows        * sizeof(         int));
	int          * y        = (         int *)malloc(   nrows        * sizeof(         int));
	int          * solution = (         int *)malloc(   nrows        * sizeof(         int));

	// Create a sparse matrix with nnz non zeros per row constant.
	// The non zero location and values are set randomly
	for (unsigned int i = 0 ; i < nrows ; ++i) {
		unsigned int start = rand() % nrows / nnz;
		columns[i] = start;
		values [i] = rand() % 100;
	}

	for (unsigned int j = 1 ; j < nnz ; ++j) {
		for (unsigned int i = 0 ; i < nrows ; ++i) {

			columns[j * nrows + i] = columns[(j-1) * nrows + i] + rand() % nrows / nnz + 1;
			values [j * nrows + i] = rand() % 100;
		}
	}

	// Create the source (x) vector
	for (unsigned int i = 0; i < nrows; ++i)
		x[i] = 1;

	// Perform a matrix-vector multiply: y = A*x
	// Very inefficient implementation here
	for (unsigned int i = 0; i < nrows; ++i) {
		int sum = 0;
		for (unsigned int j=0; j<nnz; ++j) {
			unsigned int offset = j * nrows + i;
			sum += values[offset] * x[columns[offset]];
		}
		solution[i] = sum;
	}

	unsigned int *d_columns;
	check_cuda( cudaMalloc(&d_columns,   nrows * nnz * sizeof(unsigned int)) );
	int *d_values;
	check_cuda( cudaMalloc(&d_values ,   nrows * nnz * sizeof(         int)) );
	int *d_x;
	check_cuda( cudaMalloc(&d_x      ,   nrows       * sizeof(         int)) );
	int *d_y;
	check_cuda( cudaMalloc(&d_y      ,   nrows       * sizeof(         int)) );

	check_cuda( cudaMemcpy ( d_columns, columns,   nrows * nnz * sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_values , values ,   nrows * nnz * sizeof(         int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_x      , x      ,   nrows       * sizeof(         int), cudaMemcpyHostToDevice ) );

	for (unsigned int i = 0; i < 5; ++i)
		gSpMatVecMulELLInt( d_columns, d_values , d_x , d_y , nrows, nnz ) ;

	check_cuda( cudaMemcpy ( y        , d_y    ,   nrows       * sizeof(         int), cudaMemcpyDeviceToHost ) );

	for (unsigned int j = 0; j < nrows ; ++j) {
		if (  solution[j] != y[j] ) {
			std::cout << "Values are different !" << std::endl;
		}
	}

	free(columns);
	free(values);
	free(x);
	free(y);
	return 0;
}
