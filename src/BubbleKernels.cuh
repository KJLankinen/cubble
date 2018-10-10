#pragma once

#include <cuda_runtime_api.h>
#include "Vec.h"
#include "UtilityKernels.cuh"

namespace cubble
{
extern __device__ int dMaxBubblesPerCell;
extern __device__ int dNumPairs;
extern __device__ double dTotalFreeArea;
extern __device__ double dTotalFreeAreaPerRadius;
extern __device__ double dVolumeMultiplier;
extern __device__ double dTotalVolume;

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ double getWrappedCoordinate(double val1, double val2, double multiplier);
__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb, dvec tfr, ivec cellDim);
__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);

template <typename... Args>
__device__ void comparePair(int idx1, int idx2, double *r, int *first, int *second, Args... args)
{
    const double radii = r[idx1] + r[idx2];
    if (getDistanceSquared(idx1, idx2, args...) < 1.5 * radii * radii)
    {
        // Set the smaller idx to idx1 and larger to idx2
        int id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        id = atomicAdd(&dNumPairs, 1);
        first[id] = idx1;
        second[id] = idx2;
    }
}

__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue);
template <typename... Args>
__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue, Args... args)
{
    wrapAround(idx, coordinate, minValue, maxValue);
    wrapAround(idx, args...);
}

__device__ void addVelocity(int idx1, int idx2, double multiplier, double maxDistance, double minDistance, bool shouldWrap, double *x, double *v);
template <typename... Args>
__device__ void addVelocity(int idx1, int idx2, double multiplier, double maxDistance, double minDistance, bool shouldWrap, double *x, double *v, Args... args)
{
    addVelocity(idx1, idx2, multiplier, maxDistance, minDistance, shouldWrap, x, v);
    addVelocity(idx1, idx2, multiplier, args...);
}

template <typename... Args>
__device__ void forceBetweenPair(int idx1, int idx2, double fZeroPerMuZero, double *r, Args... args)
{
    const double radii = r[idx1] + r[idx2];
    double multiplier = getDistanceSquared(idx1, idx2, args...);
    if (radii * radii >= multiplier)
    {
        multiplier = sqrt(multiplier);
        multiplier = fZeroPerMuZero * (radii - multiplier) / (radii * multiplier);
        addVelocity(idx1, idx2, multiplier, args...);
    }
}

__device__ void forceFromWalls(int idx, double fZeroPerMuZero, double *r,
                               double interval, double zeroPoint, bool shouldWrap, double *x, double *v);
template <typename... Args>
__device__ void forceFromWalls(int idx, double fZeroPerMuZero, double *r,
                               double interval, double zeroPoint, bool shouldWrap, double *x, double *v,
                               Args... args)
{
    forceFromWalls(idx, fZeroPerMuZero, r, interval, zeroPoint, shouldWrap, x, v);
    forceFromWalls(idx, fZeroPerMuZero, r, args...);
}

template <typename... Args>
__global__ void boundaryWrapKernel(int numValues, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
        wrapAround(tid, args...);
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

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec lbb, dvec tfr, ivec cellDim, int numBubbles);

template <typename... Args>
__global__ void neighborSearch(int neighborCellNumber,
                               int numValues,
                               int numCells,
                               int numMaxPairs,
                               int *offsets,
                               int *sizes,
                               int *first,
                               int *second,
                               double *r,
                               Args... args)
{
    const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
    const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
    const int cellIdx2 = getNeighborCellIndex(idxVec, dimVec, neighborCellNumber);

    if (cellIdx2 >= 0)
    {
        const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);
        DEVICE_ASSERT(cellIdx1 < numCells);
        DEVICE_ASSERT(cellIdx2 < numCells);

        if (sizes[cellIdx1] == 0 || sizes[cellIdx2] == 0)
            return;

        // Self comparison only loops the upper triangle of values (n * (n - 1)) / 2 comparisons instead of n^2.
        if (cellIdx1 == cellIdx2)
        {
            const int size = sizes[cellIdx1];
            const int offset = offsets[cellIdx1];
            for (int k = threadIdx.x; k < (size * (size - 1)) / 2; k += blockDim.x)
            {
                int idx1 = size - 2 - (int)floor(sqrt(-8.0 * k + 4 * size * (size - 1) - 7) * 0.5 - 0.5);
                const int idx2 = offset + k + idx1 + 1 - size * (size - 1) / 2 + (size - idx1) * ((size - idx1) - 1) / 2;
                idx1 += offset;

                DEVICE_ASSERT(idx1 < numValues);
                DEVICE_ASSERT(idx2 < numValues);
                DEVICE_ASSERT(idx1 != idx2);

                comparePair(idx1, idx2, r, first, second, args...);
            }
        }
        else // Compare all values of one cell to all values of other cell, resulting in n1 * n2 comparisons.
        {
            const int size1 = sizes[cellIdx1];
            const int size2 = sizes[cellIdx2];
            const int offset1 = offsets[cellIdx1];
            const int offset2 = offsets[cellIdx2];
            for (int k = threadIdx.x; k < size1 * size2; k += blockDim.x)
            {
                const int idx1 = offset1 + k / size2;
                const int idx2 = offset2 + k % size2;

                DEVICE_ASSERT(idx1 < numValues);
                DEVICE_ASSERT(idx2 < numValues);
                DEVICE_ASSERT(idx1 != idx2);

                comparePair(idx1, idx2, r, first, second, args...);
            }
        }
    }
}

template <typename... Args>
__global__ void velocityKernel(int numValues, double fZeroPerMuZero, int *first, int *second, double *r, Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
        forceBetweenPair(first[i], second[i], fZeroPerMuZero, r, args...);

#if (PBC_X != 1 || PBC_Y != 1 || PBC_Z != 1)
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
        forceFromWalls(i, fZeroPerMuZero, r, args...);
#endif
}

template <typename... Args>
__global__ void potentialEnergyKernel(int numValues,
                                      int *first,
                                      int *second,
                                      double *r,
                                      double *energy,
                                      Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
    {
        const int idx1 = first[i];
        const int idx2 = second[i];
        double e = r[idx1] + r[idx2] - sqrt(getDistanceSquared(idx1, idx2, args...));
        e *= e;
        atomicAdd(&energy[idx1], e);
        atomicAdd(&energy[idx2], e);
    }
}

template <typename... Args>
__global__ void gasExchangeKernel(int numValues,
                                  double pi,
                                  int *first,
                                  int *second,
                                  double *r,
                                  double *drdt,
                                  double *freeArea,
                                  Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
    {
        const int idx1 = first[i];
        const int idx2 = second[i];

        const double magnitude = sqrt(getDistanceSquared(idx1, idx2, args...));
        const double r1 = r[idx1];
        const double r2 = r[idx2];

        if (magnitude <= r1 + r2)
        {
            double overlapArea = 0;

            if (magnitude < r1 || magnitude < r2)
            {
                overlapArea = r1 < r2 ? r1 : r2;
                overlapArea *= overlapArea;
            }
            else
            {
                overlapArea = 0.5 * (r2 * r2 - r1 * r1 + magnitude * magnitude) / magnitude;
                overlapArea *= overlapArea;
                overlapArea = r2 * r2 - overlapArea;
                DEVICE_ASSERT(overlapArea > -0.0001);
                overlapArea = overlapArea < 0 ? -overlapArea : overlapArea;
                DEVICE_ASSERT(overlapArea >= 0);
            }
#if (NUM_DIM == 3)
            overlapArea *= pi;
#else
            overlapArea = 2.0 * sqrt(overlapArea);
#endif
            atomicAdd(&freeArea[idx1], overlapArea);
            atomicAdd(&freeArea[idx2], overlapArea);

            overlapArea *= (1.0 / r2 - 1.0 / r1);

            atomicAdd(&drdt[idx1], overlapArea);
            atomicAdd(&drdt[idx2], -overlapArea);
        }
    }
}

__global__ void freeAreaKernel(int numValues, double pi, double *r, double *freeArea, double *freeAreaPerRadius);

__global__ void finalRadiusChangeRateKernel(double *drdt,
                                            double *r,
                                            double *freeArea,
                                            int numBubbles,
                                            double invPi,
                                            double kappa,
                                            double kParam);

__global__ void addVolume(double *r, int numBubbles);

__global__ void calculateRedistributedGasVolume(double *volume,
                                                double *r,
                                                int *aboveMinRadFlags,
                                                double pi,
                                                int numBubbles);
} // namespace cubble
