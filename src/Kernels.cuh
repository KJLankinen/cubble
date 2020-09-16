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
__global__ void bubblesToCells(int *cellIndices, int *bubbleIndices,
                               ivec cellDim, Bubbles bubbles);
__device__ void comparePair(int idx1, int idx2, Bubbles &bubbles, Pairs &pairs);
__global__ void neighborSearch(int neighborCellNumber, int numCells,
                               int *offsets, int *sizes, Bubbles bubbles,
                               Pairs pairs);
__global__ void reorganizeByIndex(Bubbles bubbles, const int *newIndex);
__global__ void pairVelocity(Bubbles bubbles, Pairs pairs);
__global__ void wallVelocity(Bubbles bubbles);
__global__ void averageNeighborVelocity(Bubbles bubbles, Pairs pairs);
__global__ void imposedFlowVelocity(Bubbles bubbles);
__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs);
__global__ void pairwiseGasExchange(Bubbles bubbles, Pairs pairs);
__global__ void mediatedGasExchange(Bubbles bubbles);
__global__ void predict(double timeStep, bool useGasExchange, Bubbles bubbles);
__global__ void correct(double timeStep, bool useGasExchange, Bubbles bubbles);
__global__ void incrementPath(Bubbles bubbles);
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs);
__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs);
__global__ void euler(double timeStep, Bubbles bubbles);
__global__ void transformPositions(bool normalize, Bubbles bubbles);
__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles);
__global__ void calculateVolumes(Bubbles bubbles);
__global__ void assignDataToBubbles(ivec bubblesPerDim, double avgRad,
                                    Bubbles bubbles);
__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);
__device__ int getGlobalTid();
__device__ dvec wrappedDifference(dvec p1, dvec p2, dvec interval);
__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim);
__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);
__device__ __host__ unsigned int encodeMorton2(unsigned int x, unsigned int y);
__device__ __host__ unsigned int encodeMorton3(unsigned int x, unsigned int y,
                                               unsigned int z);
__device__ __host__ unsigned int decodeMorton2x(unsigned int code);
__device__ __host__ unsigned int decodeMorton2y(unsigned int code);
__device__ __host__ unsigned int decodeMorton3x(unsigned int code);
__device__ __host__ unsigned int decodeMorton3y(unsigned int code);
__device__ __host__ unsigned int decodeMorton3z(unsigned int code);
__device__ __host__ unsigned int part1By1(unsigned int x);
__device__ __host__ unsigned int part1By2(unsigned int x);
__device__ __host__ unsigned int compact1By1(unsigned int x);
__device__ __host__ unsigned int compact1By2(unsigned int x);

template <typename... Arguments>
void cudaLaunch(const char *kernelNameStr, const char *file, int line,
                void (*f)(Arguments...), KernelSize kernelSize,
                uint32_t sharedMemBytes, cudaStream_t stream,
                Arguments... args) {
#ifndef NDEBUG
    assertMemBelowLimit(kernelNameStr, file, line, sharedMemBytes);
    assertBlockSizeBelowLimit(kernelNameStr, file, line, kernelSize.block);
    assertGridSizeBelowLimit(kernelNameStr, file, line, kernelSize.grid);
#endif

    f<<<kernelSize.grid, kernelSize.block, sharedMemBytes, stream>>>(args...);

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
__device__ void resetArrayToValue(T value, int idx, T *array) {
    array[idx] = value;
}
template <typename T, typename... Args>
__device__ void resetArrayToValue(T value, int idx, T *array, Args... args) {
    resetArrayToValue(value, idx, array);
    resetArrayToValue(value, idx, args...);
}

template <typename T, typename... Args>
__global__ void resetKernel(T value, int numValues, Args... args) {
    const int tid = getGlobalTid();
    if (tid < numValues)
        resetArrayToValue(value, tid, args...);

    if (tid == 0) {
        dTotalArea = 0.0;
        dTotalOverlapArea = 0.0;
        dTotalOverlapAreaPerRadius = 0.0;
        dTotalAreaPerRadius = 0.0;
        dTotalVolumeNew = 0.0;
        dMaxError = 0.0;
        dMaxRadius = 0.0;
        dMaxExpansion = 0.0;
        dNumToBeDeleted = 0;
        dNumPairsNew = dNumPairs;
    }
}
} // namespace cubble
