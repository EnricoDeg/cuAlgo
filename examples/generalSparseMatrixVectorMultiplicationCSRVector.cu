/*
 * @file generalSparseMatrixVectorMultiplicationCSRVector.cu
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
	const unsigned int nnz   = 128 ;

	// Allocate enough storage for the matix.  We allocate more than
	// is needed in order to simplify the code
	unsigned int * columns  = (unsigned int *)malloc(   nrows * nnz  * sizeof(unsigned int));
	int          * values   = (         int *)malloc(   nrows * nnz  * sizeof(         int));
	unsigned int * row_ptr  = (unsigned int *)malloc( ( nrows + 1 )  * sizeof(unsigned int));
	int          * x        = (         int *)malloc(   nrows        * sizeof(         int));
	int          * y        = (         int *)malloc(   nrows        * sizeof(         int));
	int          * solution = (         int *)malloc(   nrows        * sizeof(         int));

	// Create a sparse matrix with nnz non zeros per row constant.
	// The non zero location and values are set randomly
	for (unsigned int i = 0 ; i < nrows ; ++i) {

		int start = rand() % nrows / nnz;
		columns[i * nnz] = start;
		values [i * nnz] = rand() % 100;
		for (unsigned int j = 1 ; j < nnz ; ++j) {

			columns[i * nnz + j] = columns[i * nnz + j - 1] + rand() % nrows / nnz + 1;
			values [i * nnz + j] = rand() % 100;
		}
		row_ptr[i] = i * nnz ;
	}
	row_ptr[nrows] = row_ptr[nrows-1] + nnz ;

	// Create the source (x) vector
	for (unsigned int i = 0; i < nrows; ++i)
		x[i] = 1;

	// sanity check
	for (unsigned int i = 0; i < nrows; ++i)
		for (unsigned int idx=row_ptr[i]; idx<row_ptr[i+1]; ++idx)
			if (columns[idx] > nrows)
				printf("column = %d\n", columns[idx]);

	for (unsigned int i = 0; i < nrows; ++i)
		for (unsigned int idx=row_ptr[i]; idx<row_ptr[i+1]; ++idx)
			if (idx > nrows*nnz-1)
				printf("idx = %d\n", idx);

	// Perform a matrix-vector multiply: y = A*x
	for (unsigned int i = 0; i < nrows; ++i) {
		int sum = 0;
		for (unsigned int idx=row_ptr[i]; idx<row_ptr[i+1]; ++idx)
			sum += values[idx] * x[columns[idx]];
		solution[i] = sum;
	}

	unsigned int *d_columns;
	check_cuda( cudaMalloc(&d_columns,   nrows * nnz * sizeof(unsigned int)) );
	int *d_values;
	check_cuda( cudaMalloc(&d_values ,   nrows * nnz * sizeof(         int)) );
	unsigned int *d_row_ptr;
	check_cuda( cudaMalloc(&d_row_ptr, ( nrows + 1 ) * sizeof(unsigned int)) );
	int *d_x;
	check_cuda( cudaMalloc(&d_x      ,   nrows       * sizeof(         int)) );
	int *d_y;
	check_cuda( cudaMalloc(&d_y      ,   nrows       * sizeof(         int)) );

	check_cuda( cudaMemcpy ( d_columns, columns,   nrows * nnz * sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_values , values ,   nrows * nnz * sizeof(         int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_row_ptr, row_ptr, ( nrows + 1 ) * sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	check_cuda( cudaMemcpy ( d_x      , x      ,   nrows       * sizeof(         int), cudaMemcpyHostToDevice ) );

	for (int i = 0; i < 5; ++i)
		gSpMatVecMulCSRVectorInt( d_columns, d_row_ptr, d_values , d_x , d_y , nrows ) ;

	check_cuda( cudaMemcpy ( y        , d_y    ,   nrows       * sizeof(         int), cudaMemcpyDeviceToHost ) );

	for (int j = 0; j < nrows ; ++j) {
		if (  solution[j] != y[j] ) {
			std::cout << "Values are different !" << std::endl;
		}
	}

	free(columns);
	free(values);
	free(row_ptr);
	free(x);
	free(y);
	return 0;
}
