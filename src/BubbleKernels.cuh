#pragma once

#include <cuda_runtime_api.h>
#include "Vec.h"
#include "UtilityKernels.cuh"
#include "Globals.h"

namespace cubble
{
extern __device__ int dMaxBubblesPerCell;
extern __device__ int dNumPairs;
extern __device__ CubbleFloatType dTotalFreeArea;
extern __device__ CubbleFloatType dTotalFreeAreaPerRadius;
extern __device__ CubbleFloatType dVolumeMultiplier;
extern __device__ CubbleFloatType dTotalVolume;
extern __device__ CubbleFloatType dInvRho;
extern __device__ CubbleFloatType dTotalArea;
extern __device__ CubbleFloatType dAverageSurfaceAreaIn;

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ CubbleFloatType getWrappedCoordinate(CubbleFloatType val1, CubbleFloatType val2, CubbleFloatType multiplier);
__device__ int getCellIdxFromPos(CubbleFloatType x, CubbleFloatType y, CubbleFloatType z, fpvec lbb, fpvec tfr, ivec cellDim);
__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);

template <typename... Args>
__device__ void comparePair(int idx1, int idx2, CubbleFloatType *r, int *first, int *second, Args... args)
{
    const CubbleFloatType radii = r[idx1] + r[idx2];
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

__device__ void wrapAround(int idx, CubbleFloatType *coordinate, CubbleFloatType minValue, CubbleFloatType maxValue);
template <typename... Args>
__device__ void wrapAround(int idx, CubbleFloatType *coordinate, CubbleFloatType minValue, CubbleFloatType maxValue, Args... args)
{
    wrapAround(idx, coordinate, minValue, maxValue);
    wrapAround(idx, args...);
}

__device__ void addVelocity(int idx1, int idx2, CubbleFloatType multiplier, CubbleFloatType maxDistance, CubbleFloatType minDistance, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v);
template <typename... Args>
__device__ void addVelocity(int idx1, int idx2, CubbleFloatType multiplier, CubbleFloatType maxDistance, CubbleFloatType minDistance, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v, Args... args)
{
    addVelocity(idx1, idx2, multiplier, maxDistance, minDistance, shouldWrap, x, v);
    addVelocity(idx1, idx2, multiplier, args...);
}

template <typename... Args>
__device__ void forceBetweenPair(int idx1, int idx2, CubbleFloatType fZeroPerMuZero, CubbleFloatType *r, Args... args)
{
    const CubbleFloatType radii = r[idx1] + r[idx2];
    CubbleFloatType multiplier = getDistanceSquared(idx1, idx2, args...);
    if (radii * radii >= multiplier)
    {
        multiplier = sqrt(multiplier);
        multiplier = fZeroPerMuZero * (radii - multiplier) / (radii * multiplier);
        addVelocity(idx1, idx2, multiplier, args...);
    }
}

__device__ void forceFromWalls(int idx, CubbleFloatType fZeroPerMuZero, CubbleFloatType *r,
                               CubbleFloatType interval, CubbleFloatType zeroPoint, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v);
template <typename... Args>
__device__ void forceFromWalls(int idx, CubbleFloatType fZeroPerMuZero, CubbleFloatType *r,
                               CubbleFloatType interval, CubbleFloatType zeroPoint, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v,
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

__global__ void calculateVolumes(CubbleFloatType *r, CubbleFloatType *volumes, int numValues, CubbleFloatType pi);

__global__ void assignDataToBubbles(CubbleFloatType *x,
                                    CubbleFloatType *y,
                                    CubbleFloatType *z,
                                    CubbleFloatType *xPrd,
                                    CubbleFloatType *yPrd,
                                    CubbleFloatType *zPrd,
                                    CubbleFloatType *r,
                                    CubbleFloatType *w,
                                    int *aboveMinRadFlags,
                                    ivec bubblesPerDim,
                                    fpvec tfr,
                                    fpvec lbb,
                                    CubbleFloatType avgRad,
                                    CubbleFloatType minRad,
                                    CubbleFloatType pi,
                                    int numValues);

__global__ void assignBubblesToCells(CubbleFloatType *x, CubbleFloatType *y, CubbleFloatType *z, int *cellIndices, int *bubbleIndices, fpvec lbb, fpvec tfr, ivec cellDim, int numValues);

template <typename... Args>
__global__ void neighborSearch(int neighborCellNumber,
                               int numValues,
                               int numCells,
                               int numMaxPairs,
                               int *offsets,
                               int *sizes,
                               int *first,
                               int *second,
                               CubbleFloatType *r,
                               Args... args)
{
    const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
    const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
    const int cellIdx2 = getNeighborCellIndex(idxVec, dimVec, neighborCellNumber);

    if (cellIdx2 >= 0)
    {
        const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);
        DEVICE_ASSERT(cellIdx1 < numCells, "Invalid cell index!");
        DEVICE_ASSERT(cellIdx2 < numCells, "Invalid cell index!");

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

                DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

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

                DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

                comparePair(idx1, idx2, r, first, second, args...);
            }
        }
    }
}

template <typename... Args>
__global__ void velocityPairKernel(CubbleFloatType fZeroPerMuZero, int *first, int *second, CubbleFloatType *r, Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
        forceBetweenPair(first[i], second[i], fZeroPerMuZero, r, args...);
}

template <typename... Args>
__global__ void velocityWallKernel(int numValues, CubbleFloatType fZeroPerMuZero, int *first, int *second, CubbleFloatType *r, Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
        forceFromWalls(i, fZeroPerMuZero, r, args...);
}

template <typename... Args>
__global__ void potentialEnergyKernel(int numValues,
                                      int *first,
                                      int *second,
                                      CubbleFloatType *r,
                                      CubbleFloatType *energy,
                                      Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
    {
        const int idx1 = first[i];
        const int idx2 = second[i];
        CubbleFloatType e = r[idx1] + r[idx2] - sqrt(getDistanceSquared(idx1, idx2, args...));
        e *= e;
        atomicAdd(&energy[idx1], e);
        atomicAdd(&energy[idx2], e);
    }
}

template <typename... Args>
__global__ void gasExchangeKernel(int numValues,
                                  CubbleFloatType pi,
                                  int *first,
                                  int *second,
                                  CubbleFloatType *r,
                                  CubbleFloatType *drdt,
                                  CubbleFloatType *freeArea,
                                  Args... args)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs; i += gridDim.x * blockDim.x)
    {
        const int idx1 = first[i];
        const int idx2 = second[i];

        const CubbleFloatType magnitude = sqrt(getDistanceSquared(idx1, idx2, args...));
        const CubbleFloatType r1 = r[idx1];
        const CubbleFloatType r2 = r[idx2];

        if (magnitude < r1 + r2)
        {
            CubbleFloatType overlapArea = 0;

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
                DEVICE_ASSERT(overlapArea > -0.0001, "Overlap area is negative!");
                overlapArea = overlapArea < 0 ? -overlapArea : overlapArea;
                DEVICE_ASSERT(overlapArea >= 0, "Overlap area is negative!");
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

__global__ void freeAreaKernel(int numValues, CubbleFloatType pi, CubbleFloatType *r, CubbleFloatType *relativeFreeArea, CubbleFloatType *relativeFreeAreaPerRadius, CubbleFloatType *area);

__global__ void finalRadiusChangeRateKernel(CubbleFloatType *drdt,
                                            CubbleFloatType *r,
                                            CubbleFloatType *relativeFreeArea,
                                            int numValues,
                                            CubbleFloatType invPi,
                                            CubbleFloatType kappa,
                                            CubbleFloatType kParam);

__global__ void addVolume(CubbleFloatType *r, int numValues);

__global__ void calculateRedistributedGasVolume(CubbleFloatType *volume,
                                                CubbleFloatType *r,
                                                int *aboveMinRadFlags,
                                                CubbleFloatType pi,
                                                int numValues);
} // namespace cubble
