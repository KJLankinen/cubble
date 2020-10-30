/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "DataDefinitions.h"
#include "Util.h"
#include "Vec.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cubble {
extern __device__ Constants *dConstants;
extern __device__ double dTotalArea;
extern __device__ double dTotalOverlapArea;
extern __device__ double dTotalOverlapAreaPerRadius;
extern __device__ double dTotalAreaPerRadius;
extern __device__ double dTotalVolumeNew;
extern __device__ double dMaxRadius;
extern __device__ bool dErrorEncountered;
extern __device__ int dNumPairs;
extern __device__ int dNumPairsNew;
extern __device__ int dNumToBeDeleted;
}; // namespace cubble

namespace cubble {
__global__ void preIntegrate(double ts, bool useGasExchange, Bubbles bubbles,
                             double *temp1, double *temp2);
__global__ void pairwiseInteraction(Bubbles bubbles, Pairs pairs,
                                    double *overlap, bool useGasExchange,
                                    bool useFlow, bool stabilize);
__global__ void postIntegrate(double ts, bool useGasExchange,
                              bool incrementPath, bool useFlow, bool stabilize,
                              Bubbles bubbles, double *maximums,
                              double *overlap, int *toBeDeleted);
__device__ void addFlowVelocity(Bubbles &bubbles, int i);
__device__ void addWallVelocity(Bubbles &bubbles, int i);
__global__ void cellByPosition(int *cellIndices, int *cellSizes, ivec cellDim,
                               Bubbles bubbles);
__global__ void indexByCell(int *cellIndices, int *cellOffsets,
                            int *bubbleIndices, int count);
__device__ void comparePair(int idx1, int idx2, int *histogram, int *pairI,
                            int *pairJ, Bubbles &bubbles, Pairs &pairs);
__global__ void neighborSearch(int numCells, int numNeighborCells, ivec cellDim,
                               int *offsets, int *sizes, int *histogram,
                               int *pairI, int *pairJ, Bubbles bubbles,
                               Pairs pairs);
__global__ void sortPairs(Bubbles bubbles, Pairs pairs, int *pairI, int *pairJ);
__global__ void countNumNeighbors(Bubbles bubbles, Pairs pairs);
__global__ void reorganizeByIndex(Bubbles bubbles, const int *newIndex);
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int *toBeDeleted);
__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int *toBeDeleted);
__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs, double *energy);
__global__ void euler(double ts, Bubbles bubbles);
__global__ void transformPositions(bool normalize, Bubbles bubbles);
__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles);
__global__ void calculateVolumes(Bubbles bubbles, double *volumes);
__global__ void assignDataToBubbles(ivec bubblesPerDim, double avgRad,
                                    Bubbles bubbles);
__global__ void initGlobals();
__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);
__device__ dvec wrappedDifference(double x1, double y1, double z1, double x2,
                                  double y2, double z2);
__device__ int getNeighborCellIndex(int cellIdx, ivec dim, int neighborNum);
__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim);
__device__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);

template <typename T> __device__ T sum(T a, T b) { return a + b; }
template <typename T> __device__ T max(T a, T b) { return a > b ? a : b; }
template <typename T> __device__ void reduce(T *addr, int warp, T (*f)(T, T)) {
    // Assumes that addr.length() == BLOCK_SIZE
    const int tid = threadIdx.x;
    const int wid = (tid & 31);
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE / 32; i++) {
        if (i == warp) {
            continue;
        }
        addr[tid] = f(addr[tid], addr[wid + i * 32]);
    }
    __syncwarp();

    T temp;
    temp = addr[tid ^ 16];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 8];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 4];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 2];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    if (0 == wid) {
        addr[tid] = f(addr[tid], addr[tid + 1]);
    }
}

template <typename... Arguments>
void cubLaunch(const char *file, int line,
               cudaError_t (*func)(void *, size_t &, Arguments...),
               void *tempMem, uint64_t maxMem, Arguments... args) {
    uint64_t tempMemReq = 0;
    (*func)(NULL, tempMemReq, args...);
    if (tempMemReq > maxMem) {
        std::stringstream ss;
        ss << "Not enough temporary memory for cub function call @" << file
           << ":" << line << ".\nRequested " << tempMemReq
           << " bytes, maximum is " << maxMem << " bytes.";
        throw std::runtime_error(ss.str());
    }
    (*func)(tempMem, tempMemReq, args...);
}

template <typename... Arguments>
void cudaLaunch(const char *kernelNameStr, const char *file, int line,
                void (*f)(Arguments...), const Params &params,
                uint32_t sharedMemBytes, cudaStream_t stream,
                Arguments... args) {
#ifndef NDEBUG
    assertMemBelowLimit(kernelNameStr, file, line, sharedMemBytes);
    assertBlockSizeBelowLimit(kernelNameStr, file, line, params.threadBlock);
    assertGridSizeBelowLimit(kernelNameStr, file, line, params.blockGrid);
#endif

    f<<<params.blockGrid, params.threadBlock, sharedMemBytes, stream>>>(
        args...);

#ifndef NDEBUG
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaPeekAtLastError());

    bool errorEncountered = false;
    CUDA_ASSERT(cudaMemcpyFromSymbol(static_cast<void *>(&errorEncountered),
                                     dErrorEncountered, sizeof(bool)));

    if (errorEncountered) {
        std::stringstream ss;
        ss << "Error encountered during kernel execution."
           << "\nError location: '" << kernelNameStr << "' @" << file << ":"
           << line << "."
           << "\nSee earlier messages for possible details.";

        throw std::runtime_error(ss.str());
    }
#endif
}

template <typename T> __device__ void swapValues(int from, int to, T *arr) {
    arr[to] = arr[from];
}
template <typename T, typename... Args>
__device__ void swapValues(int from, int to, T *arr, Args... args) {
    swapValues(from, to, arr);
    swapValues(from, to, args...);
}

__device__ void resetDeviceGlobals();
template <typename T>
__device__ void setArrayToValue(T value, int idx, T *array) {
    array[idx] = value;
}
template <typename T, typename... Args>
__device__ void setArrayToValue(T value, int idx, T *array, Args... args) {
    setArrayToValue(value, idx, array);
    setArrayToValue(value, idx, args...);
}

template <typename T, typename... Args>
__global__ void resetArrays(T value, int numValues, bool resetGlobals,
                            Args... args) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        setArrayToValue(value, i, args...);
    }

    if (resetGlobals && threadIdx.x + blockIdx.x == 0) {
        resetDeviceGlobals();
    }
}
} // namespace cubble
