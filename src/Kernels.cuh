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
__device__ dvec getWrappedPos(dvec pos, dvec tfr, dvec lbb);
__device__ int getCellIdxFromPos(double x, double y, double z, dvec interval, ivec cellDim);

__device__ void resetDoubleArrayToValue(double value, int tid, double *array);

template <typename... Args>
__device__ void resetDoubleArrayToValue(double value, int tid, double *array, Args... args)
{
    array[tid] = value;
    resetDoubleArrayToValue(value, tid, args...);
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
    toArray[toIndex] = fromArray[fromIndex];
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
    if (flag)
    {
        toArray[toIndex] = fromArray[fromIndex];
        copyValueIfSet(fromIndex, toIndex, flag, args...);
    }
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

__global__ void predict(double *x,
                        double *y,
                        double *z,
                        double *r,

                        double *xPrd,
                        double *yPrd,
                        double *zPrd,
                        double *rPrd,

                        double *dxdt,
                        double *dydt,
                        double *dzdt,
                        double *drdt,

                        double *dxdtOld,
                        double *dydtOld,
                        double *dzdtOld,
                        double *drdtOld,

                        dvec tfr,
                        dvec lbb,
                        double timeStep,
                        int numBubbles,
                        bool useGasExchange);

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

__global__ void correct(double *x,
                        double *y,
                        double *z,
                        double *r,

                        double *xPrd,
                        double *yPrd,
                        double *zPrd,
                        double *rPrd,

                        double *dxdt,
                        double *dydt,
                        double *dzdt,
                        double *drdt,

                        double *dxdtPrd,
                        double *dydtPrd,
                        double *dzdtPrd,
                        double *drdtPrd,

                        double *errors,
                        int *aboveMinRadFlags,
                        double minRad,
                        dvec tfr,
                        dvec lbb,
                        double timeStep,
                        int numBubbles,
                        bool useGasExchange);

__global__ void addVolume(double *r, double *volumeMultiplier, int numBubbles, double invTotalVolume);

__global__ void eulerIntegration(double *x,
                                 double *y,
                                 double *z,
                                 double *r,

                                 double *dxdt,
                                 double *dydt,
                                 double *dzdt,
                                 double *drdt,

                                 dvec tfr,
                                 dvec lbb,
                                 double timeStep,
                                 int numBubbles);

__global__ void calculateRedistributedGasVolume(double *volume,
                                                double *r,
                                                int *aboveMinRadFlags,
                                                double *volumeMultiplier,
                                                double pi,
                                                int numBubbles);
} // namespace cubble