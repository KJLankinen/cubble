#pragma once

#include <cuda_runtime_api.h>
#include "UtilityKernels.cuh"

namespace cubble
{
__device__ void adamsBashforth(int idx, double timeStep, double *yNext, double *y, double *f, double *fPrevious);
template <typename... Args>
__device__ void adamsBashforth(int idx, double timeStep, double *yNext, double *y, double *f, double *fPrevious, Args... args)
{
    adamsBashforth(idx, timeStep, yNext, y, f, fPrevious);
    adamsBashforth(idx, timeStep, args...);
}

template <typename... Args>
__global__ void predictKernel(int numValues, double timeStep, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        adamsBashforth(tid, timeStep, args...);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext, double *y, double *f, double *fNext);
template <typename... Args>
__device__ double adamsMoulton(int idx, double timeStep, double *yNext, double *y, double *f, double *fNext, Args... args)
{
    double error1 = adamsMoulton(idx, timeStep, yNext, y, f, fNext);
    const double error2 = adamsMoulton(idx, timeStep, args...);
    error1 = error1 > error2 ? error1 : error2;

    return error1;
}

template <typename... Args>
__global__ void correctKernel(int numValues, double timeStep, double *errors, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        errors[tid] = adamsMoulton(tid, timeStep, args...);
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f);
template <typename... Args>
__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f, Args... args)
{
    eulerIntegrate(idx, timeStep, y, f);
    eulerIntegrate(idx, timeStep, args...);
}

template <typename... Args>
__global__ void eulerKernel(int numValues, double timeStep, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        eulerIntegrate(tid, timeStep, args...);
}
}