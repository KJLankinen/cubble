// -*- C++ -*-

#pragma once

#include <cuda_runtime_api.h>
#include "Macros.h"
#include "Vec.h"

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

struct ExecutionPolicy
{
    ExecutionPolicy(dim3 gridSize, dim3 blockSize, uint32_t sharedMemBytes, cudaStream_t stream)
        : gridSize(gridSize), blockSize(blockSize), sharedMemBytes(sharedMemBytes), stream(stream)
    {
    }

    ExecutionPolicy(uint32_t numThreadsPerBlock, uint32_t numTotalThreads)
    {
        blockSize = dim3(numThreadsPerBlock, 1, 1);
        gridSize = dim3((uint32_t)std::ceil(numTotalThreads / (float)numThreadsPerBlock), 1, 1);
        sharedMemBytes = 0;
        stream = 0;
    }

    ExecutionPolicy(uint32_t numThreadsPerBlock, uint32_t numTotalThreads, uint32_t bytes, cudaStream_t s)
    {
        blockSize = dim3(numThreadsPerBlock, 1, 1);
        gridSize = dim3((uint32_t)std::ceil(numTotalThreads / (float)numThreadsPerBlock), 1, 1);
        sharedMemBytes = bytes;
        stream = s;
    }

    dim3 gridSize;
    dim3 blockSize;
    uint32_t sharedMemBytes;
    cudaStream_t stream;
};

template <typename... Arguments>
void cudaLaunch(const ExecutionPolicy &p, void (*f)(Arguments...), Arguments... args)
{
    f<<<p.gridSize, p.blockSize, p.sharedMemBytes, p.stream>>>(args...);
#ifndef NDEBUG
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());
#endif
}

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ int getGlobalTid();
__device__ double getWrappedCoordinate(double val1, double val2, double multiplier);
__device__ int getCellIdxFromPos(double x, double y, double z, dvec interval, ivec cellDim);

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
    double e1 = adamsMoulton(idx, timeStep, yNext, y, f, fNext);
    const double e2 = adamsMoulton(idx, timeStep, args...);
    e1 = e1 > e2 ? e1 : e2;

    return e1;
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

__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue);
template <typename... Args>
__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue, Args... args)
{
    wrapAround(idx, coordinate, minValue, maxValue);
    wrapAround(idx, args...);
}

template <typename... Args>
__global__ void boundaryWrapKernel(int numValues, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        wrapAround(tid, args...);
}

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

__global__ void calculateVolumes(double *r, double *volumes, int numBubbles, double pi);

__global__ void assignDataToBubbles(double *x,
                                    double *y,
                                    double *z,
                                    double *xPrd,
                                    double *yPrd,
                                    double *zPrd,
                                    double *r,
                                    double *w,
                                    int *aboveMinRadFlags,
                                    ivec bubblesPerDim,
                                    dvec tfr,
                                    dvec lbb,
                                    double avgRad,
                                    double minRad,
                                    int numBubbles);

__global__ void findOffsets(int *cellIndices, int *offsets, int numCells, int numBubbles);
__global__ void findSizes(int *offsets, int *sizes, int numCells, int numBubbles);

__global__ void assignBubblesToCells(double *x,
                                     double *y,
                                     double *z,
                                     int *cellIndices,
                                     int *bubbleIndices,
                                     dvec interval,
                                     ivec cellDim,
                                     int numBubbles);

__global__ void findBubblePairs(double *x,
                                double *y,
                                double *z,
                                double *r,
                                int *offsets,
                                int *sizes,
                                int *firstIndices,
                                int *secondIndices,
                                int *numPairs,
                                int numCells,
                                int numBubbles,
                                dvec interval,
                                int maxNumSharedVals,
                                int maxNumPairs);

__global__ void calculateVelocityAndGasExchange(double *x,
                                                double *y,
                                                double *z,
                                                double *r,

                                                double *dxdt,
                                                double *dydt,
                                                double *dzdt,
                                                double *drdt,

                                                double *energy,
                                                double *freeArea,

                                                int *firstIndices,
                                                int *secondIndices,

                                                int numBubbles,
                                                int numPairs,
                                                double fZeroPerMuZero,
                                                double pi,
                                                dvec interval,
                                                bool calculateEnergy,
                                                bool useGasExchange);

__global__ void calculateFreeAreaPerRadius(double *r, double *freeArea, double *output, double pi, int numBubbles);

__global__ void calculateFinalRadiusChangeRate(double *drdt,
                                               double *r,
                                               double *freeArea,
                                               int numBubbles,
                                               double invRho,
                                               double invPi,
                                               double kappa,
                                               double kParam);

__global__ void addVolume(double *r, double *volumeMultiplier, int numBubbles, double invTotalVolume);

__global__ void calculateRedistributedGasVolume(double *volume,
                                                double *r,
                                                int *aboveMinRadFlags,
                                                double *volumeMultiplier,
                                                double pi,
                                                int numBubbles);
} // namespace cubble