/*
 * @file cuAlgoSupport.hpp
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
#ifndef CUALGOSUPPORT_HPP
#define CUALGOSUPPORT_HPP

namespace cuAlgo{

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
 * @ingroup algoSupport
 */
unsigned int * getRowBlocks( const unsigned int * row_ptr     ,
                                   unsigned int   nrows       ,
                                   unsigned int * blocks_count);

}

#endif