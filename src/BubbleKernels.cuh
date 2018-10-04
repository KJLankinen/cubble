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
__device__ void comparePair(int idx1, int idx2, double *r, int neighborStride, int *neighbors, Args... args)
{
    const double radii = r[idx1] + r[idx2];
    if (getDistanceSquared(idx1, idx2, args...) < 1.2 * radii * radii)
    {
        int neighborIdx = 1 + atomicAdd(&neighbors[idx1 * neighborStride], 1);
        DEVICE_ASSERT(neighborIdx < neighborStride);
        neighbors[neighbordIdx] = idx2;

        neighborIdx = 1 + atomicAdd(&neighbors[idx2 * neighborStride], 1);
        DEVICE_ASSERT(neighborIdx < neighborStride);
        neighbors[neighbordIdx] = idx1;
    }
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

template <typename... Args>
__global__ void findNeighbors(int neighborCellNumber, int neighborStride, int *neighbors, int *offsets, int *sizes, double *r, Args... args)
{
    const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
    const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
    const int cellIdx2 = getNeighborCellIndex(idxVec, dimVec, neighborCellNumber);

    if (cellIdx2 >= 0)
    {
        const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);

        // Self comparison only loops the upper triangle of values (n * (n - 1)) / 2 comparisons instead of n^2.
        if (cellIdx1 == cellIdx2)
        {
            const int size = sizes[cellIdx1];
            const int offset = offsets[cellIdx1];
            for (int k = threadIdx.x; k < (size * (size - 1)) / 2; k += blockDim.x)
            {
                int idx1 = size - 2 - floor(sqrt(-8 * k + 4 * size * (size - 1) - 7) * 0.5 - 0.5);
                const int idx2 = offset + k + idx1 + 1 - size * (size - 1) / 2 + (size - idx1) * ((size - idx1) - 1) / 2;
                idx1 += offset;
                comparePair(idx1, idx2, r, neighborStride, neighbors, args...);
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
                comparePair(idx1, idx2, r, neighborStride, neighbors, args...);
            }
        }
    }
}

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