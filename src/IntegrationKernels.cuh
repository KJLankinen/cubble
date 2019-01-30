#pragma once

#include <cuda_runtime_api.h>
#include "UtilityKernels.cuh"

namespace cubble
{
__device__ void adamsBashforth(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fPrevious);
template <typename... Args>
__device__ void adamsBashforth(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fPrevious, Args... args)
{
    adamsBashforth(idx, timeStep, yNext, y, f, fPrevious);
    adamsBashforth(idx, timeStep, args...);
}

template <typename... Args>
__global__ void predictKernel(int numValues, CubbleFloatType timeStep, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        adamsBashforth(tid, timeStep, args...);
}

__device__ CubbleFloatType adamsMoulton(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fNext);
template <typename... Args>
__device__ CubbleFloatType adamsMoulton(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fNext, Args... args)
{
    CubbleFloatType error1 = adamsMoulton(idx, timeStep, yNext, y, f, fNext);
    const CubbleFloatType error2 = adamsMoulton(idx, timeStep, args...);
    error1 = error1 > error2 ? error1 : error2;

    return error1;
}

template <typename... Args>
__global__ void correctKernel(int numValues, CubbleFloatType timeStep, CubbleFloatType *errors, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        errors[tid] = adamsMoulton(tid, timeStep, args...);
}

__device__ void eulerIntegrate(int idx, CubbleFloatType timeStep, CubbleFloatType *y, CubbleFloatType *f);
template <typename... Args>
__device__ void eulerIntegrate(int idx, CubbleFloatType timeStep, CubbleFloatType *y, CubbleFloatType *f, Args... args)
{
    eulerIntegrate(idx, timeStep, y, f);
    eulerIntegrate(idx, timeStep, args...);
}

template <typename... Args>
__global__ void eulerKernel(int numValues, CubbleFloatType timeStep, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        eulerIntegrate(tid, timeStep, args...);
}
}