// -*- C++ -*-

#pragma once

#include <cuda_runtime_api.h>
#include "Macros.h"
#include "Vec.h"

namespace cubble
{
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

__global__ void resetDoubleArrayToValue(double *array, double value, int numValues);

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
                                    int givenNumBubblesPerDim,
                                    dvec tfr,
                                    dvec lbb,
                                    double avgRad,
                                    double minRad,
                                    int numBubbles);

__global__ void calculateOffsets(double *x,
                                 double *y,
                                 double *z,
                                 int *sizes,
                                 dvec domainDim,
                                 int numBubbles,
                                 int numCells);

__global__ void bubblesToCells(double *x,
                               double *y,
                               double *z,
                               int *indices,
                               int *offsets,
                               int *sizes,
                               dvec domainDim,
                               int numBubbles);

__global__ void findBubblePairs(double *x,
                                double *y,
                                double *z,
                                double *r,
                                int *indices,
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

__global__ void removeSmallBubbles(double *x,
                                   double *y,
                                   double *z,
                                   double *r,

                                   double *xTemp,
                                   double *yTemp,
                                   double *zTemp,
                                   double *rTemp,

                                   double *dxdt,
                                   double *dydt,
                                   double *dzdt,
                                   double *drdt,

                                   double *dxdtTemp,
                                   double *dydtTemp,
                                   double *dzdtTemp,
                                   double *drdtTemp,

                                   double *dxdtOld,
                                   double *dydtOld,
                                   double *dzdtOld,
                                   double *drdtOld,

                                   double *dxdtOldTemp,
                                   double *dydtOldTemp,
                                   double *dzdtOldTemp,
                                   double *drdtOldTemp,

                                   int *newIdx,
                                   int *flag,
                                   int numBubbles);
} // namespace cubble