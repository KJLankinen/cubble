#pragma once

#include "Util.h"
#include "Vec.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cubble {
extern __device__ double dTotalArea;
extern __device__ double dTotalOverlapArea;
extern __device__ double dTotalOverlapAreaPerRadius;
extern __device__ double dTotalAreaPerRadius;
extern __device__ double dTotalVolume;
extern __device__ bool dErrorEncountered;
extern __device__ int dNumPairs;
extern __device__ int dNumBubblesAboveMinRad;
extern __device__ double dVolumeMultiplier;

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
        dTotalVolume = 0.0;
        dVolumeMultiplier = 0.0;
        dNumBubblesAboveMinRad = 0;
    }
}

template <typename T>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray,
                          T *toArray) {
    toArray[toIndex] = fromArray[fromIndex];
}

template <typename T, typename... Args>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray, T *toArray,
                          Args... args) {
    copyValue(fromIndex, toIndex, fromArray, toArray);
    copyValue(fromIndex, toIndex, args...);
}

template <typename... Args>
__global__ void copyKernel(int numValues, ReorganizeType reorganizeType,
                           int *indices, int *flags, Args... args) {
    const int tid = getGlobalTid();
    if (tid < numValues) {
        bool copy = true;
        int from, to;

        switch (reorganizeType) {
        case ReorganizeType::COPY_FROM_INDEX:
            from = indices[tid];
            to = tid;
            break;
        case ReorganizeType::COPY_TO_INDEX:
            from = tid;
            to = indices[tid];
            break;
        case ReorganizeType::CONDITIONAL_FROM_INDEX:
            from = indices[tid];
            to = tid;
            copy = 1 == flags[tid];
            break;
        case ReorganizeType::CONDITIONAL_TO_INDEX:
            from = tid;
            to = indices[tid];
            copy = 1 == flags[tid];
            break;
        default:
            break;
        }

        if (copy) {
            copyValue(from, to, args...);
        }
    }
}

__device__ dvec wrappedDifference(dvec p1, dvec p2, dvec interval);

__global__ void transformPositionsKernel(bool normalize, int numValues,
                                         dvec lbb, dvec tfr, double *x,
                                         double *y, double *z);

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb,
                                 dvec tfr, ivec cellDim);
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

__device__ void comparePair(int idx1, int idx2, double *r, int *first,
                            int *second, dvec interval, double *x, double *y,
                            double *z);

__global__ void wrapKernel(int numValues, dvec lbb, dvec tfr, double *x,
                           double *y, double *z, int *mx, int *my, int *mz);

__global__ void calculateVolumes(double *r, double *volumes, int numValues);

__global__ void assignDataToBubbles(double *x, double *y, double *z,
                                    double *xPrd, double *yPrd, double *zPrd,
                                    double *r, double *w, int *indices,
                                    ivec bubblesPerDim, dvec tfr, dvec lbb,
                                    double avgRad, double minRad,
                                    int numValues);

__global__ void assignBubblesToCells(double *x, double *y, double *z,
                                     int *cellIndices, int *bubbleIndices,
                                     dvec lbb, dvec tfr, ivec cellDim,
                                     int numValues);

__global__ void neighborSearch(int neighborCellNumber, int numValues,
                               int numCells, int numMaxPairs, int *offsets,
                               int *sizes, int *first, int *second, double *r,
                               dvec interval, double *x, double *y, double *z);

__global__ void velocityPairKernel(double fZeroPerMuZero, int *pair1,
                                   int *pair2, double *r, dvec interval,
                                   double *x, double *y, double *z, double *vx,
                                   double *vy, double *vz);

__global__ void velocityWallKernel(int numValues, double *r, double *x,
                                   double *y, double *z, double *vx, double *vy,
                                   double *vz, dvec lbb, dvec tfr,
                                   double fZeroPerMuZero, double dragCoeff);

__global__ void neighborVelocityKernel(int *first, int *second,
                                       int *numNeighbors, double *sumX,
                                       double *sumY, double *sumZ, double *vx,
                                       double *vy, double *vz);

__global__ void flowVelocityKernel(int numValues, int *numNeighbors,
                                   double *velX, double *velY, double *velZ,
                                   double *nVelX, double *nVelY, double *nVelZ,
                                   double *posX, double *posY, double *posZ,
                                   double *r, dvec flowVel, dvec flowTfr,
                                   dvec flowLbb);

__global__ void potentialEnergyKernel(int numValues, int *first, int *second,
                                      double *r, double *energy, dvec interval,
                                      double *x, double *y, double *z);

__global__ void gasExchangeKernel(int numValues, int *pair1, int *pair2,
                                  dvec interval, double *r, double *drdt,
                                  double *freeArea, double *x, double *y,
                                  double *z);

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r,
                                            double *freeArea, int numValues,
                                            double kappa, double kParam,
                                            double averageSurfaceAreaIn);

__global__ void addVolume(double *r, int numValues);

__global__ void predictKernel(int numValues, double timeStep,
                              bool useGasExchange, double *xn, double *x,
                              double *vx, double *vxp, double *yn, double *y,
                              double *vy, double *vyp, double *zn, double *z,
                              double *vz, double *vzp, double *rn, double *r,
                              double *vr, double *vrp);

__global__ void correctKernel(int numValues, double timeStep,
                              bool useGasExchange, double minRad,
                              double *errors, double *maxR, int *flags,
                              double *xp, double *x, double *vx, double *vxp,
                              double *yp, double *y, double *vy, double *vyp,
                              double *zp, double *z, double *vz, double *vzp,
                              double *rp, double *r, double *vr, double *vrp);

__global__ void endStepKernel(int numValues, double *errors, double *maxR,
                              int origBlockSize);

__global__ void eulerKernel(int numValues, double timeStep, double *x,
                            double *vx, double *y, double *vy, double *z,
                            double *vz);

__global__ void
pathLengthDistanceKernel(int numValues, dvec interval, double *pathLength,
                         double *pathLengthPrev, double *squaredDistance,
                         double *x, double *xPrev, double *x0, int *wrapCountX,
                         double *y, double *yPrev, double *y0, int *wrapCountY,
                         double *z, double *zPrev, double *z0, int *wrapCountZ);

} // namespace cubble
