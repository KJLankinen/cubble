#pragma once

#include <cuda_runtime_api.h>
#include "Vec.h"
#include "UtilityKernels.cuh"

namespace cubble
{
extern __device__ int dMaxBubblesPerCell;
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
__device__ void comparePair(int idx1, int idx2, int neighborStride, int numValues, double *r, int *neighbors, Args... args)
{
    const double radii = r[idx1] + r[idx2];
    if (getDistanceSquared(idx1, idx2, args...) < 1.2 * radii * radii)
    {
        int row = 1 + atomicAdd(&neighbors[idx1], 1);
        DEVICE_ASSERT(row < neighborStride + 1);
        neighbors[idx1 + row * numValues] = idx2;

        row = 1 + atomicAdd(&neighbors[idx2], 1);
        DEVICE_ASSERT(row < neighborStride + 1);
        neighbors[idx2 + row * numValues] = idx1;
    }
}

__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue);
template <typename... Args>
__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue, Args... args)
{
    wrapAround(idx, coordinate, minValue, maxValue);
    wrapAround(idx, args...);
}

__device__ void addVelocity(int idx1, int idx2, double multiplier, bool shouldReset, double maxDistance, bool shouldWrap, double *x, double *v);
template <typename... Args>
__device__ void addVelocity(int idx1, int idx2, double multiplier, bool shouldReset, double maxDistance, bool shouldWrap, double *x, double *v, Args... args)
{
    addVelocity(idx1, idx2, multiplier, shouldReset, maxDistance, shouldWrap, x, v);
    addVelocity(idx1, idx2, multiplier, shouldReset, args...);
}

template <typename... Args>
__device__ void forceBetweenPair(int idx1, int idx2, double fZeroPerMuZero, bool shouldReset, double *r, Args... args)
{
    const double radii = r[idx1] + r[idx2];
    double multiplier = sqrt(getDistanceSquared(idx1, idx2, args...));
    multiplier = fZeroPerMuZero * (radii - multiplier) / (radii * multiplier);
    addVelocity(idx1, idx2, multiplier, shouldReset, args...);
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

__global__ void findOffsets(int *cellIndices, int *offsets, int numCells, int numBubbles);
__global__ void findSizes(int *offsets, int *sizes, int numCells, int numBubbles);

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec lbb, dvec tfr, ivec cellDim, int numBubbles);

template <typename... Args>
__global__ void neighborSearch(int neighborCellNumber,
                               int neighborStride,
                               int numValues,
                               int numCells,
                               int *neighbors,
                               int *offsets,
                               int *sizes,
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

                comparePair(idx1, idx2, neighborStride, numValues, r, neighbors, args...);
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
                const int idx1 = offset1 + k / size1;
                const int idx2 = offset2 + k % size1;

                DEVICE_ASSERT(idx1 < numValues);
                DEVICE_ASSERT(idx2 < numValues);

                comparePair(idx1, idx2, neighborStride, numValues, r, neighbors, args...);
            }
        }
    }
}

template <typename... Args>
__global__ void velocityKernel(int numValues, double fZeroPerMuZero, int neighborStride, int *neighbors, double *r, Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
    {
        for (int i = 1; i <= neighbors[tid]; ++i)
            forceBetweenPair(tid, neighbors[tid + neighborStride * i], fZeroPerMuZero, i == 1, r, args...);
    }
}

template <typename... Args>
__global__ void potentialEnergyKernel(int numValues, int neighborStride, int *neighbors, double *r, double *energy, Args... args)
{
    // Note: This doesn't take into account the potential energy stored in a bubble that's pressed against a wall.
    const int tid = getGlobalTid();
    if (tid < numValues)
    {
        energy[tid] = 0;
        for (int i = 1; i <= neighbors[tid]; ++i)
        {
            const int idx2 = neighbors[tid + neighborStride * i];
            const double e = r[tid] + r[idx2] - sqrt(getDistanceSquared(tid, idx2, args...));
            energy[tid] += e * e;
        }
    }
}

template <typename... Args>
__global__ void gasExchangeKernel(int numValues,
                                  double pi,
                                  int neighborStride,
                                  int *neighbors,
                                  double *r,
                                  double *drdt,
                                  double *freeArea,
                                  double *freeAreaPerRadius,
                                  Args... args)
{
    const int tid = getGlobalTid();
    if (tid < numValues)
    {
        double totalOverlapArea = 0;
        drdt[tid] = 0;
        const double r1 = r[tid];
        for (int i = 1; i <= neighbors[tid]; ++i)
        {
            const int idx2 = neighbors[tid + neighborStride * i];
            const double magnitude = sqrt(getDistanceSquared(tid, idx2, args...));
            const double r2 = r[idx2];

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
            totalOverlapArea += overlapArea;
            drdt[tid] += overlapArea * (1.0 / r2 - 1.0 / r1);
        }

        double area = 2.0 * pi * r1;
#if (NUM_DIM == 3)
        area *= 2.0 * r1;
#endif
        freeArea[tid] = area - totalOverlapArea;
        freeAreaPerRadius[tid] = freeArea[tid] / r[tid];
    }
}

__global__ void calculateFinalRadiusChangeRate(double *drdt,
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