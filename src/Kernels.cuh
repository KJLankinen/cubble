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
    void *dee = nullptr;
    CUDA_ASSERT(cudaGetSymbolAddress((void **)&dee, dErrorEncountered));
    if (dee != nullptr)
        CUDA_ASSERT(cudaMemcpy(static_cast<void *>(&errorEncountered), dee,
                               sizeof(bool), cudaMemcpyDeviceToHost));
    else
        throw std::runtime_error(
            "Couldn't get symbol address for dErrorEncountered variable!");

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
__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);
__device__ int getGlobalTid();

template <typename T> __device__ void swapValues(int from, int to, T *arr) {
    arr[to] = arr[from];
}
template <typename T, typename... Args>
__device__ void swapValues(int from, int to, T *arr, Args... args) {
    swapValues(from, to, arr);
    swapValues(from, to, args...);
}

__device__ void resetDoubleArrayToValue(double value, int idx, double *array);
template <typename... Args>
__device__ void resetDoubleArrayToValue(double value, int idx, double *array,
                                        Args... args) {
    resetDoubleArrayToValue(value, idx, array);
    resetDoubleArrayToValue(value, idx, args...);
}

template <typename... Args>
__global__ void resetKernel(double value, int numValues, Args... args) {
    const int tid = getGlobalTid();
    if (tid < numValues)
        resetDoubleArrayToValue(value, tid, args...);

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

__device__ dvec wrappedDifference(dvec p1, dvec p2, dvec interval);
__global__ void transformPositionsKernel(bool normalize, Bubbles bubbles);
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
__device__ void comparePair(int idx1, int idx2, Bubbles &bubbles, Pairs &pairs);
__global__ void wrapKernel(Bubbles bubbles);
__global__ void calculateVolumes(Bubbles bubbles);
__global__ void assignDataToBubbles(ivec bubblesPerDim, double avgRad,
                                    Bubbles bubbles);
__global__ void assignBubblesToCells(int *cellIndices, int *bubbleIndices,
                                     ivec cellDim, Bubbles bubbles);
__global__ void neighborSearch(int neighborCellNumber, int numCells,
                               int *offsets, int *sizes, Bubbles bubbles,
                               Pairs pairs);
__global__ void velocityPairKernel(Bubbles bubbles, Pairs pairs);
__global__ void velocityWallKernel(Bubbles bubbles);
__global__ void neighborVelocityKernel(Bubbles bubbles, Pairs pairs);
__global__ void flowVelocityKernel(Bubbles bubbles);
__global__ void potentialEnergyKernel(Bubbles bubbles, Pairs pairs);
__global__ void gasExchangeKernel(Bubbles bubbles, Pairs pairs);
__global__ void finalRadiusChangeRateKernel(Bubbles bubbles);
__global__ void predictKernel(double timeStep, bool useGasExchange,
                              Bubbles bubbles);
__global__ void correctKernel(double timeStep, bool useGasExchange,
                              Bubbles bubbles);
__global__ void endStepKernel(int origBlockSize, Bubbles bubbles);
__global__ void eulerKernel(double timeStep, Bubbles bubbles);
__global__ void incrementPath(Bubbles bubbles);
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs);
__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs);
__global__ void reorganizeByIndex(Bubbles bubbles, const int *newIndex);
} // namespace cubble
