/*
 * @file batchNormFwdTraining.hpp
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

#include <type_traits>
#include "cuAlgo/internals/definitions.hpp"
#include "cuAlgo/internals/utils.hpp"

template<
bool layoutNHWC,
typename Type,
typename TypeAccum
>
CUALGO_DEVICE
inline TypeAccum loadFromStash(Type* stash,
    unsigned int vindex,
    unsigned int ygroupoffset,
    unsigned int ystride,
    unsigned int xgrp_sz,
    unsigned int xgrp_id,
    unsigned int xlid,
    unsigned int xstride)
{
    unsigned int index;

    if constexpr (std::is_same<Type, TypeAccum>::value) {
        index = (ygroupoffset + vindex) * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
    } else {
        if constexpr (layoutNHWC) {
            index = (ygroupoffset + vindex * 2 + xlid % 2) * ystride +
                (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
        } else {
            index = (ygroupoffset + vindex * 2) * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
        }
    }
    return *((TypeAccum*)(stash + index));
}

template<
bool layoutNHWC,
typename Type,
typename TypeAccum
>
CUALGO_DEVICE
inline void storeToStash(TypeAccum value,
    Type* stash,
    unsigned int vindex,
    unsigned int ygroupoffset,
    unsigned int ystride,
    unsigned int xgrp_sz,
    unsigned int xgrp_id,
    unsigned int xlid,
    unsigned int xstride)
{
    unsigned int index;

    if constexpr (std::is_same<Type, TypeAccum>::value) {
        index = (ygroupoffset + vindex) * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
    } else {
        if constexpr (layoutNHWC) {
            index = (ygroupoffset + vindex * 2 + xlid % 2) * ystride +
                (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
        } else {
            index = (ygroupoffset + vindex * 2) * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
        }
    }
    *((TypeAccum*)(stash + index)) = value;
}

template<typename T>
CUALGO_DEVICE
inline void blockReduceShMem2d(T* x,
                               T* y,
                               T scale,
                               T* lcl_data_x,
                               T* lcl_data_y,
                               unsigned int lid,
                               unsigned int size) {

    lcl_data_x[lid] = (T)*x;
    lcl_data_y[lid] = (T)*y;
    __syncthreads();
    for(unsigned int red = (size >> 1); red > 0; red >>= 1)
    {
        if(lid < red)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        __syncthreads();
    }
    *x = (T)(lcl_data_x[0] * scale);
    *y = (T)(lcl_data_y[0] * scale);
}

template<
unsigned int MaxBlockSize,
bool layoutNHWC,
typename Type,
typename TypeAccum>
CUALGO_GLOBAL
void batchNormFwdTrainKernel1(const Type * CUALGO_RESTRICT in,
                              Type * CUALGO_RESTRICT mvbuff,
                              unsigned int N,
                              unsigned int C,
                              unsigned int HW) {

    unsigned int xgid      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid      = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int xlid      = threadIdx.x;
    unsigned int ylid      = threadIdx.y;
    unsigned int xgrp_id   = blockIdx.x;
    unsigned int ygrp_id   = blockIdx.y;
    unsigned int xgrp_sz   = blockDim.x;
    unsigned int ygrp_sz   = blockDim.y;
    unsigned int index;
    unsigned int xstride   = layoutNHWC ? 1 : HW;
    unsigned int ystride   = layoutNHWC ? C : 1;
    unsigned int CHW       = C * HW;

    TypeAccum mean      = (TypeAccum)0.;
    TypeAccum variance  = (TypeAccum)0.;
    TypeAccum value;

    if(xgid >= C)
        return;

    if(ygid < HW)
    {
        Type read4;
        for(unsigned int n = 0; n < N; n++)
        {
            index = n * CHW + ygid * ystride + xgid * xstride;
            read4 = *((const Type*)(in + index));
            value = TypeAccum(read4);
            mean += value;
            variance = fma(value, value, variance);
        }
    }

    CUALGO_SHMEM TypeAccum lcl_data_x[MaxBlockSize];
    CUALGO_SHMEM TypeAccum lcl_data_y[MaxBlockSize];
    blockReduceShMem2d(&mean,
                   &variance,
                   (TypeAccum)1.0,
                   lcl_data_x + xlid * ygrp_sz,
                   lcl_data_y + xlid * ygrp_sz,
                   ylid,
                   ygrp_sz);

    if(ylid == 0)
    {
        storeToStash<layoutNHWC>(
            mean,
            (Type*)mvbuff,
            0U,
            ygrp_sz * ygrp_id,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        storeToStash<layoutNHWC>(
            variance,
            (Type*)mvbuff,
            1U,
            ygrp_sz * ygrp_id,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
    }
}

template<
unsigned int MaxBlockSize,
bool layoutNHWC,
typename Type,
typename TypeAccum>
CUALGO_GLOBAL
void batchNormFwdTrainKernel2(Type * CUALGO_RESTRICT meanvarbuff,
                              TypeAccum INHW,
                              double epsilon,
                              unsigned int NGRPS,
                              unsigned int N,
                              unsigned int C,
                              unsigned int HW) {

    unsigned int xgid      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int xlid      = threadIdx.x;
    unsigned int ylid      = threadIdx.y;
    unsigned int xgrp_id   = blockIdx.x;
    unsigned int xgrp_sz   = blockDim.x;
    unsigned int ygrp_sz   = blockDim.y;
    unsigned int xstride   = layoutNHWC ? 1 : HW;
    unsigned int ystride   = layoutNHWC ? C : 1;

    TypeAccum mean      = (TypeAccum)0.;
    TypeAccum variance  = (TypeAccum)0.;
    TypeAccum value, invVariance;

    if(xgid >= C)
        return;

    for(unsigned int yoffset = ylid; yoffset < NGRPS; yoffset += ygrp_sz)
    {
        mean += loadFromStash<layoutNHWC, Type, TypeAccum>(
            (Type*)meanvarbuff,
            0U,
            ygrp_sz * yoffset,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        variance += loadFromStash<layoutNHWC, Type, TypeAccum>(
            (Type*)meanvarbuff,
            1U,
            ygrp_sz * yoffset,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
    }

    CUALGO_SHMEM TypeAccum lcl_data_x[MaxBlockSize];
    CUALGO_SHMEM TypeAccum lcl_data_y[MaxBlockSize];
    blockReduceShMem2d(&mean,
                       &variance,
                       (TypeAccum)INHW,
                       lcl_data_x + xlid * ygrp_sz,
                       lcl_data_y + xlid * ygrp_sz,
                       ylid,
                       ygrp_sz);

    variance = fma(-mean, mean, variance);
    variance = max(variance, (TypeAccum)0.);
    invVariance = rsqrt(variance + (TypeAccum)epsilon);

    for(unsigned int yoffset = ylid; yoffset < NGRPS; yoffset += ygrp_sz)
    {
        storeToStash<layoutNHWC>(
            mean,
            (Type*)meanvarbuff,
            0U,
            ygrp_sz * yoffset,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        storeToStash<layoutNHWC>(
            invVariance,
            (Type*)meanvarbuff,
            1U,
            ygrp_sz * yoffset,
            ystride,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
    }
}

template<
unsigned int MaxBlockSize,
bool layoutNHWC,
unsigned int WarpSize,
typename Type,
typename TypeAccum>
CUALGO_GLOBAL
void batchNormFwdTrainKernel3(const Type * CUALGO_RESTRICT in,
                              Type * CUALGO_RESTRICT out,
                              TypeAccum* scale,
                              TypeAccum* bias,
                              unsigned int N,
                              unsigned int C,
                              unsigned int HW) {

    unsigned int xgid      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid      = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int xlid      = threadIdx.x;
    unsigned int ylid      = threadIdx.y;
    unsigned int xgrp_id   = blockIdx.x;
    unsigned int ygrp_id   = blockIdx.y;
    unsigned int xgrp_sz   = blockDim.x;
    unsigned int ygrp_sz   = blockDim.y;
    unsigned int index;
    unsigned int xstride   = layoutNHWC ? 1 : HW;
    unsigned int ystride   = layoutNHWC ? C : 1;
    unsigned int CHW       = C * HW;

    TypeAccum mean         = (TypeAccum)0.;
    TypeAccum invVariance  = (TypeAccum)0.;
    TypeAccum inhat        = (TypeAccum)0.;
    TypeAccum pvt_scale    = (TypeAccum)0.;
    TypeAccum pvt_bias     = (TypeAccum)0.;
    TypeAccum value;
    CUALGO_SHMEM TypeAccum lcl_bias[WarpSize];
    CUALGO_SHMEM TypeAccum lcl_scale[WarpSize];
    CUALGO_SHMEM TypeAccum lcl_mean[WarpSize];
    CUALGO_SHMEM TypeAccum lcl_ivar[WarpSize];

    if(xgid >= C)
        return;

    if(ylid == 0)
    {
        lcl_scale[xlid] = *((TypeAccum*)(scale + xgid));
        lcl_bias[xlid]  = *((TypeAccum*)(bias  + xgid));
        lcl_mean[xlid]  =
            loadFromStash<layoutNHWC, Type, TypeAccum>(
                (Type*)out,
                0U,
                ygrp_sz * ygrp_id,
                ystride,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        lcl_ivar[xlid] =
            loadFromStash<layoutNHWC, Type, TypeAccum>(
                (Type*)out,
                1U,
                ygrp_sz * ygrp_id,
                ystride,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
    }
    __syncthreads();

    if(ygid < HW)
    {
        mean        = lcl_mean[xlid];
        invVariance = lcl_ivar[xlid];
        pvt_scale   = lcl_scale[xlid];
        pvt_bias    = lcl_bias[xlid];

        Type read4;
        for(unsigned int n = 0; n < N; n++)
        {
            index = n * CHW + ygid * ystride + xgid * xstride;
            value = *((Type*)(in + index));
            inhat = TypeAccum(value);
            inhat = (inhat - mean) * invVariance;
            inhat = fma(pvt_scale, inhat, pvt_bias);
            value = Type(inhat);
            *((Type*)(out + index)) = value;
        }
    }
}

namespace cuAlgo {

    template<
    unsigned int MaxBlockSize,
    unsigned int WarpSize,
    bool layoutNHWC,
    typename T>
    void batchNormFwdTraining(const T *in,
                              T *out,
                              float* scale,
                              float* bias,
                              unsigned int N,
                              unsigned int C,
                              unsigned int HW,
                              cudaStream_t stream = 0,
                              bool async = false) {

        unsigned int Cp2 = uint(1 << int(std::ceil(std::log2(C))));
        unsigned int BlockSizeX =
            layoutNHWC ? std::min(Cp2, WarpSize) : 1;
        unsigned int BlockSizeY =
            layoutNHWC ? MaxBlockSize / BlockSizeX : MaxBlockSize;

        dim3 threadsPerBlock(BlockSizeX, BlockSizeY);
        dim3 blocksPerGrid(div_ceil(C, BlockSizeX),
                           div_ceil(HW, BlockSizeY));

        print_kernel_config(threadsPerBlock, blocksPerGrid);
        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async, 
              CUALGO_KERNEL_NAME(batchNormFwdTrainKernel1<MaxBlockSize, layoutNHWC, T, float>),
              in, out, N, C, HW);

        dim3 singleBlockPerGrid(1, 1);

        print_kernel_config(threadsPerBlock, singleBlockPerGrid);
        TIME( singleBlockPerGrid, threadsPerBlock, 0, stream, async, 
              CUALGO_KERNEL_NAME(batchNormFwdTrainKernel2<MaxBlockSize, layoutNHWC, T, float>),
              out, (float)1.0 / (N * HW), (double)1.0e-7, blocksPerGrid.y, N, C, HW);

        print_kernel_config(threadsPerBlock, blocksPerGrid);
        TIME( blocksPerGrid, threadsPerBlock, 0, stream, async, 
              CUALGO_KERNEL_NAME(batchNormFwdTrainKernel3<MaxBlockSize, layoutNHWC, WarpSize, T, float>),
              in, out, scale, bias, N, C, HW);
    }
}
