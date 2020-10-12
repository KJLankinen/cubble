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
extern __device__ double dMaxError;
extern __device__ double dMaxRadius;
extern __device__ double dMaxExpansion;
extern __device__ bool dErrorEncountered;
extern __device__ int dNumPairs;
extern __device__ int dNumPairsNew;
extern __device__ int dNumToBeDeleted;
}; // namespace cubble

namespace cubble {
__global__ void cellByPosition(int *cellIndices, int *cellSizes, ivec cellDim,
                               Bubbles bubbles);
__global__ void indexByCell(int *cellIndices, int *cellOffsets,
                            int *bubbleIndices, int count);
__device__ void comparePair(int idx1, int idx2, int *histogram, int *pairI,
                            int *pairJ, Bubbles &bubbles);
__global__ void neighborSearch(int numCells, int numNeighborCells, ivec cellDim,
                               int *offsets, int *sizes, int *histogram,
                               int *pairI, int *pairJ, Bubbles bubbles);
__global__ void sortPairs(Bubbles bubbles, Pairs pairs, int *pairI, int *pairJ);
__global__ void countNumNeighbors(Bubbles bubbles, Pairs pairs);
__global__ void reorganizeByIndex(Bubbles bubbles, const int *newIndex);
__global__ void pairVelocity(Bubbles bubbles, Pairs pairs);
__global__ void wallVelocity(Bubbles bubbles);
__global__ void averageNeighborVelocity(Bubbles bubbles, Pairs pairs);
__global__ void imposedFlowVelocity(Bubbles bubbles);
__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs, double *energy);
__global__ void pairwiseGasExchange(Bubbles bubbles, Pairs pairs,
                                    double *overlap);
__global__ void mediatedGasExchange(Bubbles bubbles, double *overlap);
__global__ void predict(double timeStep, bool useGasExchange, Bubbles bubbles);
__global__ void correct(double timeStep, bool useGasExchange, Bubbles bubbles,
                        double *maximums, int *toBeDeleted);
__global__ void incrementPath(Bubbles bubbles);
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int *toBeDeleted);
__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int *toBeDeleted);
__global__ void euler(double timeStep, Bubbles bubbles);
__global__ void transformPositions(bool normalize, Bubbles bubbles);
__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles);
__global__ void calculateVolumes(Bubbles bubbles, double *volumes);
__global__ void assignDataToBubbles(ivec bubblesPerDim, double avgRad,
                                    Bubbles bubbles);
__global__ void initGlobals();
__device__ int atomicAggInc(int *ctr);
__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);
__device__ int getGlobalTid();
__device__ dvec wrappedDifference(dvec interval, double x1, double y1,
                                  double z1, double x2, double y2, double z2);
__device__ int getNeighborCellIndex(int cellIdx, ivec dim, int neighborNum);
__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim);
__device__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);

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

template <typename T>
__device__ void addTo(int idx, int mul, T *to, T *from, bool predicate) {
    if (predicate) {
        *to += mul * from[idx];
    }
}
template <typename T>
__device__ void addTo(int idx, int mul, T *to, T *from, T *, bool predicate) {
    addTo(idx, mul, to, from, predicate);
}
template <typename T, typename... Args>
__device__ void addTo(int idx, int mul, T *to, T *from, T *, bool predicate,
                      Args... args) {
    addTo(idx, mul, to, from, predicate);
    addTo(idx, mul, args...);
}

template <typename T>
__device__ void recursiveAtomicAdd(T *addr, T *from, bool predicate) {
    if (predicate) {
        atomicAdd(addr, *from);
    }
}
template <typename T>
__device__ void recursiveAtomicAdd(T *from, T *, T *addr, bool predicate) {
    recursiveAtomicAdd(addr, from, predicate);
}
template <typename T, typename... Args>
__device__ void recursiveAtomicAdd(T *from, T *, T *addr, bool predicate,
                                   Args... args) {
    recursiveAtomicAdd(addr, from, predicate);
    recursiveAtomicAdd(args...);
}

template <typename... Args>
__device__ void warpCoopAdd(int matchOn, Args... args) {
    const unsigned int active = __activemask();
    __syncwarp(active);
    const unsigned int matches = __match_any_sync(active, matchOn);
    const unsigned int lanemask_lt = (1 << (threadIdx.x & 31)) - 1;
    const unsigned int rank = __popc(matches & lanemask_lt);

    if (0 == rank) {
        const int flt = 32 * (threadIdx.x >> 5);
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int mul = !!(matches & 1 << j);
            addTo(j + flt, mul, args...);
        }

        recursiveAtomicAdd(args...);
    }
    __syncwarp(active);
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
