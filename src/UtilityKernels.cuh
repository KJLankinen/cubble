#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "Macros.h"
#include "Util.h"
#include "assert.h"

namespace cubble
{
enum class ReorganizeType
{
    COPY_FROM_INDEX,
    COPY_TO_INDEX,
    CONDITIONAL_FROM_INDEX,
    CONDITIONAL_TO_INDEX,

    NUM_VALUES
};

extern __device__ bool dErrorEncountered;

template <typename... Arguments>
void cudaLaunch(const char *kernelNameStr, const char *file, int line, void (*f)(Arguments...), const ExecutionPolicy &p, Arguments... args)
{
#ifndef NDEBUG
    assertMemBelowLimit(kernelNameStr, file, line, p.sharedMemBytes);
    assertBlockSizeBelowLimit(kernelNameStr, file, line, p.blockSize);
    assertGridSizeBelowLimit(kernelNameStr, file, line, p.gridSize);
#endif

    f<<<p.gridSize, p.blockSize, p.sharedMemBytes, p.stream>>>(args...);

#ifndef NDEBUG
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());

    bool errorEncountered = false;
    void *dee = nullptr;
    CUDA_CALL(cudaGetSymbolAddress((void **)&dee, dErrorEncountered));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(&errorEncountered), dee, sizeof(bool), cudaMemcpyDeviceToHost));

    if (errorEncountered)
    {
        std::stringstream ss;
        ss << "Error encountered during kernel execution."
           << "\nError location: '" << kernelNameStr << "' @" << file << ":" << line << "."
           << "\nSee earlier messages for possible details.";

        throw std::runtime_error(ss.str());
    }
#endif
}

__device__ void logError(bool condition, const char *statement, const char *errMsg);

__device__ int getGlobalTid();

__device__ void resetDoubleArrayToValue(double value, int idx, double *array);

template <typename... Args>
__device__ void resetDoubleArrayToValue(double value, int idx, double *array, Args... args)
{
    resetDoubleArrayToValue(value, idx, array);
    resetDoubleArrayToValue(value, idx, args...);
}

template <typename... Args>
__global__ void resetKernel(double value, int numValues, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        resetDoubleArrayToValue(value, tid, args...);
}

template <typename T>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray, T *toArray)
{
    toArray[toIndex] = fromArray[fromIndex];
}

template <typename T, typename... Args>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray, T *toArray, Args... args)
{
    copyValue(fromIndex, toIndex, fromArray, toArray);
    copyValue(fromIndex, toIndex, args...);
}

template <typename T>
__device__ void copyValueIfSet(int fromIndex, int toIndex, bool flag, T *fromArray, T *toArray)
{
    if (flag)
        toArray[toIndex] = fromArray[fromIndex];
}

template <typename T, typename... Args>
__device__ void copyValueIfSet(int fromIndex, int toIndex, bool flag, T *fromArray, T *toArray, Args... args)
{
    copyValueIfSet(fromIndex, toIndex, flag, fromArray, toArray);
    copyValueIfSet(fromIndex, toIndex, flag, args...);
}

template <typename... Args>
__global__ void reorganizeKernel(int numValues, ReorganizeType reorganizeType, int *indices, int *flags, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
    {
        switch (reorganizeType)
        {
        case ReorganizeType::COPY_FROM_INDEX:
            copyValue(indices[tid], tid, args...);
            break;
        case ReorganizeType::COPY_TO_INDEX:
            copyValue(tid, indices[tid], args...);
            break;
        case ReorganizeType::CONDITIONAL_FROM_INDEX:
            copyValueIfSet(indices[tid], tid, 1 == flags[tid], args...);
            break;
        case ReorganizeType::CONDITIONAL_TO_INDEX:
            copyValueIfSet(tid, indices[tid], 1 == flags[tid], args...);
            break;
        default:
            break;
        }
    }
}

// Could be generalized to accept any comparable type, but I've been too lazy to do that...
__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values, double constant);
template <typename... Args>
__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values, double constant, Args... args)
{
    setFlagIfLessThanConstant(idx, flags, values, constant);
    setFlagIfLessThanConstant(idx, args...);
}

template <typename... Args>
__global__ void setFlagIfLessThanConstantKernel(int numValues, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        setFlagIfLessThanConstant(tid, args...);
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags, double *values, double constant);
template <typename... Args>
__device__ void setFlagIfGreaterThanConstant(int idx, int *flags, double *values, double constant, Args... args)
{
    setFlagIfGreaterThanConstant(idx, flags, values, constant);
    setFlagIfGreaterThanConstant(idx, args...);
}

template <typename... Args>
__global__ void setFlagIfGreaterThanConstantKernel(int numValues, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        setFlagIfGreaterThanConstant(tid, args...);
}

__device__ double getWrappedDistance(double x1, double x2, double maxDistance, bool shouldWrap);

__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x);
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, double minDistance, bool shouldWrap, double *x, double *useless);
template <typename... Args>
__forceinline__ __device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, double minDistance, bool shouldWrap, double *x, double *useless, Args... args)
{
    double d = getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
    d += getDistanceSquared(idx1, idx2, args...);

    return d;
}
template <typename... Args>
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x, Args... args)
{
    double d = getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
    d += getDistanceSquared(idx1, idx2, args...);

    return d;
}
} // namespace cubble