/*
 * @file gMatMul.hpp
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
#include "cuAlgo/internals/utils.hpp"

using namespace nvcuda;

template <typename T, int vec_elements>
struct vecT
{
//   static_assert(sizeof(T)<=8, "cuAlgo can only have 1-4 elements");
};

template <typename T>
struct vecT<T, 1>
{
    T x;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 1 + 0) * Ncol + row + offset] = x;
    }
};

template <typename T>
struct vecT<T, 2>
{
    T x;
    T y;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 2 + 0) * Ncol + row + offset] = x;
        As[(col * 2 + 1) * Ncol + row + offset] = y;
    }
};

template <typename T>
struct vecT<T, 3>
{
    T x;
    T y;
    T z;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 3 + 0) * Ncol + row + offset] = x;
        As[(col * 3 + 1) * Ncol + row + offset] = y;
        As[(col * 3 + 2) * Ncol + row + offset] = z;
    }
};

template <typename T>
struct vecT<T, 4>
{
    T x;
    T y;
    T z;
    T w;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
        w = alpha * results[3] + beta * w;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 4 + 0) * Ncol + row + offset] = x;
        As[(col * 4 + 1) * Ncol + row + offset] = y;
        As[(col * 4 + 2) * Ncol + row + offset] = z;
        As[(col * 4 + 3) * Ncol + row + offset] = w;
    }
};

template <typename T>
struct vecT<T, 8>
{
    T x;
    T y;
    T z;
    T w;
    T x1;
    T y1;
    T z1;
    T w1;

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void store(T * results, T alpha, T beta) {
        x = alpha * results[0] + beta * x;
        y = alpha * results[1] + beta * y;
        z = alpha * results[2] + beta * z;
        w = alpha * results[3] + beta * w;
        x1 = alpha * results[4] + beta * x1;
        y1 = alpha * results[5] + beta * y1;
        z1 = alpha * results[6] + beta * z1;
        w1 = alpha * results[7] + beta * w1;
    }

    CUALGO_DEVICE
    CUALGO_FORCE_INLINE
    void load_transpose(T * As,
                        unsigned int row,
                        unsigned int col,
                        unsigned int offset,
                        unsigned int Ncol) {
        As[(col * 4 + 0) * Ncol + row + offset] = x;
        As[(col * 4 + 1) * Ncol + row + offset] = y;
        As[(col * 4 + 2) * Ncol + row + offset] = z;
        As[(col * 4 + 3) * Ncol + row + offset] = w;
        As[(col * 4 + 4) * Ncol + row + offset] = x1;
        As[(col * 4 + 5) * Ncol + row + offset] = y1;
        As[(col * 4 + 6) * Ncol + row + offset] = z1;
        As[(col * 4 + 7) * Ncol + row + offset] = w1;
    }
};

template<
unsigned int WITER,
unsigned int TSIZE,
unsigned int WSUBCOL,
typename T
>
CUALGO_DEVICE
CUALGO_FORCE_INLINE
void fill_register(T * reg, T* shmem) {

    for (unsigned int wSubIdx = 0; wSubIdx < WITER; ++wSubIdx) {
        for (unsigned int i = 0; i < TSIZE; ++i) {
            reg[wSubIdx * TSIZE + i] =
                shmem[wSubIdx * WSUBCOL + i];
        }
    }
}

template <
unsigned int BM,
unsigned int BN,
unsigned int BK,
unsigned int WM,
unsigned int WN,
unsigned int WNITER,
unsigned int TM,
unsigned int TN,
unsigned int NUM_THREADS,
typename T
>
CUALGO_GLOBAL
CUALGO_LAUNCH_BOUNDS(NUM_THREADS)
void gMatMulKernel(T alpha,
                   T * CUALGO_RESTRICT A,
                   T * CUALGO_RESTRICT B,
                   T beta,
                   T * CUALGO_RESTRICT C,
                   unsigned int M,
                   unsigned int N,
                   unsigned int K)
{
    // load 128 bit at once
    static constexpr int numberVectorElements = 16 / sizeof(T);
    using vecN = vecT<T,numberVectorElements>;

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const unsigned int warpIdx = threadIdx.x / CUALGO_WARPSIZE;
    const unsigned int warpCol = warpIdx % (BN / WN);
    const unsigned int warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr unsigned int WMITER = (WM * WN) / (CUALGO_WARPSIZE * TM * TN * WNITER);
    constexpr unsigned int WSUBM = WM / WMITER; // 64/2=32
    constexpr unsigned int WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const unsigned int threadIdxInWarp = threadIdx.x % CUALGO_WARPSIZE;
    const unsigned int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const unsigned int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // allocate space for the current blocktile in smem
    CUALGO_SHMEM T shmem[BM * BK + BK * BN];
    T * As = shmem;
    T * Bs = shmem + BM * BK;

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const unsigned int innerRowA = threadIdx.x / (BK / numberVectorElements);
    const unsigned int innerColA = threadIdx.x % (BK / numberVectorElements);
    constexpr unsigned int rowStrideA = (NUM_THREADS * numberVectorElements) / BK;
    const unsigned int innerRowB = threadIdx.x / (BN / numberVectorElements);
    const unsigned int innerColB = threadIdx.x % (BN / numberVectorElements);
    constexpr unsigned int rowStrideB = NUM_THREADS / (BN / numberVectorElements);

    // allocate thread-local cache for results in registerfile
    T threadResults[WMITER * TM * WNITER * TN] = {0};
    // register caches for As and Bs
    T regM[WMITER * TM] = {0};
    T regN[WNITER * TN] = {0};

    // outer-most loop over block tiles
    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // load GMEM -> SMEM
        // transpose A while loading it
        for (unsigned int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            vecN tmp = reinterpret_cast<vecN *>(
                &A[(innerRowA + offset) * K + innerColA * numberVectorElements])[0];
            tmp.load_transpose(As, innerRowA, innerColA, offset, BM);
        }

        for (unsigned int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<vecN *>(
                &Bs[(innerRowB + offset) * BN + innerColB * numberVectorElements])[0] =
                    reinterpret_cast<vecN *>(
                    &B[(innerRowB + offset) * N + innerColB * numberVectorElements])[0];
        }
        __syncthreads();

        // calculate per-thread results
        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // SHMEM -> registers
            fill_register<WMITER, TM, WSUBM>(regM,
                &As[(dotIdx * BM) + warpRow * WM + threadRowInWarp * TM]);
            fill_register<WNITER, TN, WSUBN>(regN,
                &Bs[(dotIdx * BN) + warpCol * WN + threadColInWarp * TN]);

            // Core computation
            for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                                    regM[wSubRowIdx * TM + resIdxM] *
                                    regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            T *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += numberVectorElements) {
                    // load C vector into registers
                    vecN tmp = reinterpret_cast<vecN *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.store(threadResults + i, alpha, beta);
                    // write back
                    reinterpret_cast<vecN *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

// MMA matrix tile dimensions.
template <
unsigned int BM,
unsigned int BN,
unsigned int BK,
unsigned int WM,
unsigned int WN,
unsigned int M_MMA,
unsigned int N_MMA,
unsigned int K_MMA,
unsigned int NUM_THREADS,
typename T,
typename U
>
CUALGO_GLOBAL
void gMatMulKernelWMMA(U alpha,
                       T * A,
                       T * B,
                       U beta,
                       U * C,
                       unsigned int  M,
                       unsigned int  N,
                       unsigned int  K)
{

    const unsigned int M_tiles = M / M_MMA;
    const unsigned int N_tiles = N / N_MMA;
    const unsigned int K_tiles = K / K_MMA;

    static constexpr unsigned int block_row_warps = BM / WM;
    static constexpr unsigned int block_col_warps = BN / WN;

    static constexpr unsigned int warp_row_tiles = WM / M_MMA;
    static constexpr unsigned int warp_col_tiles = WN / N_MMA;

    static constexpr unsigned int block_row_tiles = block_row_warps * warp_row_tiles;
    static constexpr unsigned int block_col_tiles = block_col_warps * warp_col_tiles;

    static constexpr unsigned int shmem_stride = N_MMA * block_row_tiles;
    static constexpr unsigned int shmem_offset = N_MMA * warp_row_tiles;

    static constexpr unsigned int chunk_k = BK / K_MMA;
    static constexpr unsigned int chunk_line_bytes = chunk_k * K_MMA * sizeof(T);
    static constexpr unsigned int warp_copy_bytes =  (CUALGO_WARPSIZE * 16);
    static constexpr unsigned int chunk_copy_lines_per_warp = (warp_copy_bytes / chunk_line_bytes);
    static constexpr unsigned int chunk_copy_line_lanes = (CUALGO_WARPSIZE / chunk_copy_lines_per_warp);

    static constexpr unsigned int padT_bank_conflicts = 16;

    // load 128 bit at once
    static constexpr int numberVectorElementsT = 16 / sizeof(T);
    static constexpr int numberVectorElementsU = 16 / sizeof(U);
    using vecN = vecT<T,numberVectorElementsT>;
    using vecU = vecT<U,numberVectorElementsU>;

    static constexpr unsigned int WARPS_PER_BLOCK1 = NUM_THREADS / CUALGO_WARPSIZE;

    extern __shared__ T shmem[][chunk_k * K_MMA + padT_bank_conflicts];

    // Warp and lane identification.
    unsigned int warpId = threadIdx.x / CUALGO_WARPSIZE;
    unsigned int laneId = threadIdx.x % CUALGO_WARPSIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = block_col_tiles * M_MMA;

    // This pointer is used to access the C matrix tiles this warp computes.
    U *shmem_warp_tile_ptr = (U*)&shmem[0][0] + (warpId / block_row_warps) *
        shmem_stride * N_MMA * block_row_warps + (warpId % block_row_warps) * shmem_offset;

    // This pointer is used to stream the C matrix block-wide tile to and from shared memory.
    U *shmem_warp_stream_ptr = (U*)&shmem[0][0] + warpId * shmem_stride * N_MMA;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * block_row_tiles) / N_tiles) * (block_col_tiles);
        const unsigned int block_tile_j = (block_pos * block_col_tiles) % N_tiles;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_tiles) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M_MMA * N +
            block_tile_j * N_MMA;
        U *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
        for (int i = 0; i < N_MMA; i++) {
            reinterpret_cast<vecN *>(shmem_warp_stream_ptr + shmem_stride * i)[0] = 
                (reinterpret_cast<vecN *>(src_gmem_warp_stream_ptr + N * i) + laneId)[0];
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K dimension.
        wmma::fragment<wmma::accumulator, M_MMA, N_MMA, K_MMA, U> c[warp_col_tiles][warp_row_tiles];

        // Load the C matrix tiles into fragments from shared memory.
        for (int i = 0; i < warp_col_tiles; i++) {
            for (int j = 0; j < warp_row_tiles; j++) {
                const U *tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * N_MMA + j * N_MMA;

                wmma::load_matrix_sync(c[i][j], tile_ptr, shmem_stride, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Scale the C matrix.
       for (int i = 0; i < warp_col_tiles; i++) {
            for (int j = 0; j < warp_row_tiles; j++) {
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        T *warp_ptr = (warpId < (WARPS_PER_BLOCK1/2)) ? (&A[block_tile_i * M_MMA * K] + M_MMA * K * (warpId % (WARPS_PER_BLOCK1/2)) * 2) :
                                              (&B[block_tile_j * N_MMA * K] + N_MMA * K * (warpId % (WARPS_PER_BLOCK1/2)) * 2);

        // Go through the global K dimension by a fixed step at a time.
        for (int tile_k = 0; tile_k < K_tiles; tile_k += chunk_k) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK1/2) ?
                (M_MMA * (warpId % (WARPS_PER_BLOCK1/2)) * 2) : 
                (N_MMA * (warpId % (WARPS_PER_BLOCK1/2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            T *lane_ptr = (warp_ptr + tile_k * K_MMA + (laneId / chunk_copy_line_lanes) * K);

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += laneId / chunk_copy_line_lanes;

            for(int i = 0; i < ((CUALGO_WARPSIZE/2) / chunk_copy_lines_per_warp) * 2; i++) {
                // Copy 16 bytes at once in each lane.
                (reinterpret_cast<vecN *>(
                    &shmem[shmem_idx][0]) + (laneId % chunk_copy_line_lanes))[0] =
                    (reinterpret_cast<vecN *>(lane_ptr) +  (laneId % chunk_copy_line_lanes))[0];

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K * chunk_copy_lines_per_warp;
                shmem_idx += chunk_copy_lines_per_warp;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            for (int k_step = 0; k_step < chunk_k; k_step++) {
                wmma::fragment<wmma::matrix_a, M_MMA, N_MMA, K_MMA, T, wmma::row_major> a[warp_col_tiles];
                wmma::fragment<wmma::matrix_b, M_MMA, N_MMA, K_MMA, T, wmma::col_major> b[warp_row_tiles];

                for (int i = 0; i < warp_col_tiles; i++) {
                    size_t shmem_idx_a = (warpId/block_row_warps) * M_MMA * block_row_warps + (i * M_MMA);
                    const T *tile_ptr = &shmem[shmem_idx_a][k_step * K_MMA];

                    wmma::load_matrix_sync(a[i], tile_ptr, K_MMA * chunk_k + padT_bank_conflicts);

                    for (int j = 0; j < warp_row_tiles; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (warp_row_tiles * N_MMA) * (warpId%2) + (j * N_MMA);
                            const T *tile_ptr = &shmem[shmem_idx_b][k_step * K_MMA];

                            wmma::load_matrix_sync(b[j], tile_ptr, K_MMA * chunk_k + padT_bank_conflicts);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the C fragments to shared memory.
        for (int i = 0; i < warp_col_tiles; i++) {
            for (int j = 0; j < warp_row_tiles; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                // warp are well-defined even though element indices within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                U *tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * K_MMA + j * N_MMA;

                wmma::store_matrix_sync(tile_ptr, c[i][j], shmem_stride, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // store C to gmem
        for (int i = 0; i < N_MMA; i++) {
            (reinterpret_cast<vecU *>(src_gmem_warp_stream_ptr + N * i) + laneId)[0] = 
                (reinterpret_cast<vecU *>(shmem_warp_stream_ptr + shmem_stride * i) + laneId)[0];
        }

        __syncthreads();
    }
}

namespace cuAlgo {

    /**
    * @brief   Perform general matrix-matrix multiplication
    * 
    * @details The following operation is performed
    *          C = alpha * A * B + beta * C
    * 
    * @param[in]    A     pointer to the input matrix.
    *                     The matrix has dimensions {K,M}.
    * @param[in]    B     pointer to the input matrix.
    *                     The matrix has dimensions {N,K}.
    * @param[inout] C     pointer to the output matrix.
    *                     The matrix has dimensions {N,M}.
    * @param[in]    M     non-contiguous dimension of the A and C matrices
    * @param[in]    N     contiguous dimension of the B and C matrix
    * @param[in]    K     contiguous dimension of the A matrix
    *                     non-contiguous dimension of the B matrix
    * @param[in]    alpha scalar parameter to apply to A * B
    * @param[in]    beta  scalar parameter to apply to C
    * @param[in]  stream CUDA stream where the kernels are launched.
    *                    Default is stream 0 (default stream)
    * @param[in]  async  bool to define if kernels are launched asynchronously
    *                    (without synchronization).
    *                    Default is false (device is synchronized after each kernel launched)
    * 
    * @ingroup algo
    */
    template <typename T>
    void gMatMul(T             alpha ,
                 T      *A     ,
                 T      *B     ,
                 T             beta  ,
                 T            *C     ,
                 unsigned int  M     ,
                 unsigned int  N     ,
                 unsigned int  K     ,
                 cudaStream_t  stream = 0,
                 bool          async = false) {

        static constexpr uint BM = 128;
        static constexpr uint BN = 128;
        static constexpr uint BK = 16;
        static constexpr uint WN = 64;
        static constexpr uint WM = 64;
        static constexpr uint WNITER = 4;
        static constexpr uint TM = 8;
        static constexpr uint TN = 4;
        static constexpr uint NUM_THREADS = 128;

        // create as many blocks as necessary to map all of C
        dim3 blocksPerGrid(div_ceil(N, BN), div_ceil(M, BM));
        dim3 threadsPerBlock(NUM_THREADS);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async,
              CUALGO_KERNEL_NAME(gMatMulKernel<BM,BN,BK,WM, WN, WNITER, TM, TN, NUM_THREADS, T>),
              alpha, A, B, beta, C, M, N, K);
    }

    template <typename T, typename U>
    void gMatMulWMMA(U             alpha,
                     T            *A,
                     T            *B,
                     U             beta,
                     U            *C,
                     unsigned int  M,
                     unsigned int  N,
                     unsigned int  K,
                     cudaStream_t  stream = 0,
                     bool          async = false) {

        static constexpr uint BM = 128;
        static constexpr uint BN = 128;
        static constexpr uint BK = 64;
        static constexpr uint WN = 64;
        static constexpr uint WM = 32;
        static constexpr uint M_MMA = 16;
        static constexpr uint N_MMA = 16;
        static constexpr uint K_MMA = 16;
        static constexpr uint NUM_THREADS = 256;

        static constexpr unsigned int block_col_tiles = 8;
        static constexpr unsigned int block_row_tiles = 8;
        static constexpr unsigned int chunk_k = 4;

        static constexpr unsigned int shmem_sz = std::max(sizeof(T) * (block_col_tiles * M_MMA) *
            (chunk_k * K_MMA + 16) * 2,
            M_MMA * (block_row_tiles) * N_MMA * (block_col_tiles) * sizeof(U));

        // create as many blocks as necessary to map all of C
        dim3 blocksPerGrid(div_ceil(N * M, BN * BM));
        dim3 threadsPerBlock(NUM_THREADS);
        print_kernel_config(threadsPerBlock, blocksPerGrid);

        check_cuda(cudaFuncSetAttribute(gMatMulKernelWMMA<BM, BN, BK, WM, WN, M_MMA, N_MMA, K_MMA, NUM_THREADS, T, U>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));

        TIME( blocksPerGrid, threadsPerBlock, shmem_sz, stream, async,
              CUALGO_KERNEL_NAME(gMatMulKernelWMMA<BM, BN, BK, WM, WN,  M_MMA, N_MMA, K_MMA, NUM_THREADS, T, U>),
              alpha, A, B, beta, C, M, N, K);
    }
}
