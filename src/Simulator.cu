#include "CubWrapper.h"
#include "DataDefinitions.h"
#include "Kernels.cuh"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include "nlohmann/json.hpp"
#include <cuda_profiler_api.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <nvToolsExt.h>
#include <sstream>
#include <string>
#include <vector>

namespace // anonymous
{
using namespace cubble;

// Forward declare
#if (USE_PROFILING == 1)
void startProfiling(bool start);
void stopProfiling(bool stop, bool &continueIntegration);
#endif
double calculateTotalEnergy(Params &params);
double calculateVolumeOfBubbles(Params &params);
double getSimulationBoxVolume(Params &params);
void saveSnapshotToFile(Params &params);

void updateCellsAndNeighbors(Params &params) {
    NVTX_RANGE_PUSH_A("Neighbors");
    params.hostData.numNeighborsSearched++;

    KERNEL_LAUNCH(wrapOverPeriodicBoundaries, params, 0, params.stream1,
                  params.bubbles);

    // Reset pairs arrays to zero
    uint64_t bytes = sizeof(int) * params.pairs.stride * 4;
    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.pairs.i), 0, bytes,
                              params.stream2));

    // Minimum size of cell is twice the sum of the skin and max bubble radius
    ivec cellDim = (params.hostConstants.interval /
                    (2 * (params.hostData.maxBubbleRadius +
                          params.hostConstants.skinRadius)))
                       .floor();
    cellDim.z = cellDim.z > 0 ? cellDim.z : 1;
    const int numCells = cellDim.x * cellDim.y * cellDim.z;

    int *cellOffsets = params.pairs.i;
    int *cellSizes = params.pairs.i + numCells;
    int *cellIndices = params.pairs.iCopy;
    int *bubbleIndices = params.pairs.iCopy + 1 * params.bubbles.stride;

    // Assign each bubble to a particular cell, based on the bubbles
    // position and count the total number of bubbles for each cell.
    KERNEL_LAUNCH(cellByPosition, params, 0, 0, cellIndices, cellSizes, cellDim,
                  params.bubbles);

    // Calculate the sum of bubbles in earlier cells and the current cell.
    // In other words, an inclusive sum of cell sizes.
    params.cw.scan<int *, int *>(&cub::DeviceScan::InclusiveSum, cellSizes,
                                 cellOffsets, numCells);

    // Assign an index to each bubble such that bubbles of first cell are stored
    // first in memory, bubbles of second cell are stored second, and so on.
    KERNEL_LAUNCH(indexByCell, params, 0, 0, cellIndices, cellOffsets,
                  bubbleIndices, params.bubbles.count);

    // This kernel reorganizes the data by swapping values from one array to
    // another in a 'loopy' fashion. The pointers must be updated after the
    // kernel.
    KERNEL_LAUNCH(reorganizeByIndex, params, 0, 0, params.bubbles,
                  const_cast<const int *>(bubbleIndices));
    double *swapper = params.bubbles.xp;
    params.bubbles.xp = params.bubbles.x;
    params.bubbles.x = swapper;

    swapper = params.bubbles.yp;
    params.bubbles.yp = params.bubbles.y;
    params.bubbles.y = swapper;

    swapper = params.bubbles.zp;
    params.bubbles.zp = params.bubbles.z;
    params.bubbles.z = swapper;

    swapper = params.bubbles.rp;
    params.bubbles.rp = params.bubbles.r;
    params.bubbles.r = swapper;

    swapper = params.bubbles.dxdtp;
    params.bubbles.dxdtp = params.bubbles.dxdt;
    params.bubbles.dxdt = swapper;

    swapper = params.bubbles.dydtp;
    params.bubbles.dydtp = params.bubbles.dydt;
    params.bubbles.dydt = swapper;

    swapper = params.bubbles.dzdtp;
    params.bubbles.dzdtp = params.bubbles.dzdt;
    params.bubbles.dzdt = swapper;

    swapper = params.bubbles.drdtp;
    params.bubbles.drdtp = params.bubbles.drdt;
    params.bubbles.drdt = swapper;

    // Note that the order is reverse from the order in the kernel
    swapper = params.bubbles.error;
    params.bubbles.error = params.bubbles.path;
    params.bubbles.path = params.bubbles.drdto;
    params.bubbles.drdto = params.bubbles.dzdto;
    params.bubbles.dzdto = params.bubbles.dydto;
    params.bubbles.dydto = params.bubbles.dxdto;
    params.bubbles.dxdto = params.bubbles.flowVx;
    params.bubbles.flowVx = swapper;

    int *swapperI = params.bubbles.index;
    params.bubbles.index = params.bubbles.wrapCountZ;
    params.bubbles.wrapCountZ = params.bubbles.wrapCountY;
    params.bubbles.wrapCountY = params.bubbles.wrapCountX;
    params.bubbles.wrapCountX = params.bubbles.numNeighbors;
    params.bubbles.numNeighbors = swapperI;

    int zero = 0;
    CUDA_CALL(
        cudaMemcpyToSymbol(dNumPairs, static_cast<void *>(&zero), sizeof(int)));

    int numCellsToSearch = 5;
    if (params.hostConstants.dimensionality == 3) {
        numCellsToSearch = 14;
    }

    // Use second pair as temporary memory
    int *histogram = params.pairs.j;
    KERNEL_LAUNCH(neighborSearch, params, 0, 0, numCells, numCellsToSearch,
                  cellDim, cellOffsets, cellSizes, histogram, params.bubbles,
                  params.pairs);

    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&params.pairs.count),
                                   dNumPairs, sizeof(int)));

#ifndef NDEBUG
    std::cout << "Max num pairs: " << params.pairs.stride
              << ", actual num pairs: " << params.pairs.count << std::endl;
#endif

    params.cw.scan<int *, int *>(&cub::DeviceScan::InclusiveSum, histogram,
                                 params.bubbles.numNeighbors,
                                 params.bubbles.count);

    KERNEL_LAUNCH(sortPairs, params, 0, 0, params.bubbles, params.pairs);

    // Count the number of neighbors for each bubble, i.e. the number of times
    // a bubble appears in a pair. We can reuse the temporary buffers from
    // earlier.
    int *hist1 = bubbleIndices;
    int *hist2 = cellIndices;
    params.cw.histogram<int *, int, int, int>(
        &cub::DeviceHistogram::HistogramEven, params.pairs.i, hist1,
        params.bubbles.count + 1, 0, params.bubbles.count, params.pairs.count);
    params.cw.histogram<int *, int, int, int>(
        &cub::DeviceHistogram::HistogramEven, params.pairs.j, hist2,
        params.bubbles.count + 1, 0, params.bubbles.count, params.pairs.count);

    KERNEL_LAUNCH(addArrays, params, 0, 0, params.bubbles.count,
                  const_cast<const int *>(hist1),
                  const_cast<const int *>(hist2), params.bubbles.numNeighbors);

    NVTX_RANGE_POP();
}

void deleteSmallBubbles(Params &params, int numToBeDeleted) {
    NVTX_RANGE_PUSH_A("BubbleRemoval");

    KERNEL_LAUNCH(swapDataCountPairs, params, 0, 0, params.bubbles,
                  params.pairs);

    KERNEL_LAUNCH(addVolumeFixPairs, params, 0, 0, params.bubbles,
                  params.pairs);

    params.bubbles.count -= numToBeDeleted;
    const int numBlocks =
        std::min(1024, (int)std::ceil(params.bubbles.count / 128.0));
    params.blockGrid = dim3(numBlocks, 1, 1);

    NVTX_RANGE_POP();
}

double stabilize(Params &params, int numStepsToRelax) {
    // This function integrates only the positions of the bubbles.
    // Gas exchange is not used. This is used for equilibrating the foam.
    double elapsedTime = 0.0;
    double error = 100000;
    params.hostData.energy1 = calculateTotalEnergy(params);
    const int numBlocks = params.blockGrid.x;

    for (int i = 0; i < numStepsToRelax; ++i) {
        do {
            KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count,
                          true, params.bubbles.dxdtp, params.bubbles.dydtp,
                          params.bubbles.dzdtp);

            KERNEL_LAUNCH(predict, params, 0, 0, params.hostData.timeStep,
                          false, params.bubbles);

            KERNEL_LAUNCH(pairVelocity, params, 0, 0, params.bubbles,
                          params.pairs);

            if (params.hostConstants.xWall || params.hostConstants.yWall ||
                params.hostConstants.zWall) {
                KERNEL_LAUNCH(wallVelocity, params, 0, 0, params.bubbles);
            }

            KERNEL_LAUNCH(correct, params, 0, 0, params.hostData.timeStep,
                          false, params.bubbles);

            // Reduce error
            error = params.cw.reduce<double, double *, double *>(
                &cub::DeviceReduce::Max, params.bubbles.tempDoubles2,
                numBlocks);

            if (error < params.hostData.errorTolerance &&
                params.hostData.timeStep < 0.1)
                params.hostData.timeStep *= 1.9;
            else if (error > params.hostData.errorTolerance)
                params.hostData.timeStep *= 0.5;

        } while (error > params.hostData.errorTolerance);

        // Update the current values with the calculated predictions
        double *swapper = params.bubbles.dxdto;
        params.bubbles.dxdto = params.bubbles.dxdt;
        params.bubbles.dxdt = params.bubbles.dxdtp;
        params.bubbles.dxdtp = swapper;

        swapper = params.bubbles.dydto;
        params.bubbles.dydto = params.bubbles.dydt;
        params.bubbles.dydt = params.bubbles.dydtp;
        params.bubbles.dydtp = swapper;

        swapper = params.bubbles.dzdto;
        params.bubbles.dzdto = params.bubbles.dzdt;
        params.bubbles.dzdt = params.bubbles.dzdtp;
        params.bubbles.dzdtp = swapper;

        swapper = params.bubbles.x;
        params.bubbles.x = params.bubbles.xp;
        params.bubbles.xp = swapper;

        swapper = params.bubbles.y;
        params.bubbles.y = params.bubbles.yp;
        params.bubbles.yp = swapper;

        swapper = params.bubbles.z;
        params.bubbles.z = params.bubbles.zp;
        params.bubbles.zp = swapper;

        elapsedTime += params.hostData.timeStep;

        // Reduce block maximum expansions to a global maximum
        double maxExpansion = params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Max,
            params.bubbles.tempDoubles2 + 2 * numBlocks, numBlocks);

        if (maxExpansion >= 0.5 * params.hostConstants.skinRadius) {
            updateCellsAndNeighbors(params);
            // After updateCellsAndNeighbors r is correct,
            // but rp is trash. pairVelocity always uses
            // predicted values, so copy r to rp
            uint64_t bytes = params.bubbles.stride * sizeof(double);
            CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                                      static_cast<void *>(params.bubbles.r),
                                      bytes, cudaMemcpyDeviceToDevice, 0));
        }
    }

    params.hostData.energy2 = calculateTotalEnergy(params);

    return elapsedTime;
}

bool integrate(Params &params) {
    NVTX_RANGE_PUSH_A("Integration function");

    double error = 100000;
    uint32_t numLoopsDone = 0;
    double *hMaxRadius = static_cast<double *>(params.pinnedMemory);
    double *hMaxExpansion = hMaxRadius + 1;
    int *hNumToBeDeleted = reinterpret_cast<int *>(hMaxExpansion + 1);
    const int numBlocks = params.blockGrid.x;

    if (params.hostData.addFlow) {
        // Average neighbor velocity is calculated from velocities of previous
        // step
        KERNEL_LAUNCH(resetArrays, params, 0, params.stream1, 0.0,
                      params.bubbles.count, false, params.bubbles.flowVx,
                      params.bubbles.flowVy, params.bubbles.flowVz);
        KERNEL_LAUNCH(averageNeighborVelocity, params, 0, params.stream1,
                      params.bubbles, params.pairs);
    }

    do {
        NVTX_RANGE_PUSH_A("Integration step");
        KERNEL_LAUNCH(resetArrays, params, 0, params.stream2, 0.0,
                      params.bubbles.count, true, params.bubbles.dxdtp,
                      params.bubbles.dydtp, params.bubbles.dzdtp,
                      params.bubbles.drdtp, params.bubbles.tempDoubles,
                      params.bubbles.tempDoubles2);

        KERNEL_LAUNCH(predict, params, 0, params.stream1,
                      params.hostData.timeStep, true, params.bubbles);
        CUDA_CALL(cudaEventRecord(params.event1, params.stream1));

        // Gas exchange can start immediately after predict, since they
        // are computed in the same stream
        KERNEL_LAUNCH(pairwiseGasExchange, params, 0, params.stream1,
                      params.bubbles, params.pairs);
        KERNEL_LAUNCH(mediatedGasExchange, params, 0, params.stream1,
                      params.bubbles);

        // Velocity calculations
        // Wait for the event recorded after predict kernel
        CUDA_CALL(cudaStreamWaitEvent(params.stream2, params.event1, 0));
        KERNEL_LAUNCH(pairVelocity, params, 0, params.stream2, params.bubbles,
                      params.pairs);

        if (params.hostData.addFlow) {
            KERNEL_LAUNCH(imposedFlowVelocity, params, 0, params.stream2,
                          params.bubbles);
        }

        if (params.hostConstants.xWall || params.hostConstants.yWall ||
            params.hostConstants.zWall) {
            KERNEL_LAUNCH(wallVelocity, params, 0, params.stream2,
                          params.bubbles);
        }

        KERNEL_LAUNCH(correct, params, 0, 0, params.hostData.timeStep, true,
                      params.bubbles);

        // Reduce error
        error = params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Max, params.bubbles.tempDoubles2, numBlocks);

        if (error < params.hostData.errorTolerance &&
            params.hostData.timeStep < 0.1)
            params.hostData.timeStep *= 1.9;
        else if (error > params.hostData.errorTolerance)
            params.hostData.timeStep *= 0.5;

        ++numLoopsDone;
        NVTX_RANGE_POP();
    } while (error > params.hostData.errorTolerance);

    CUDA_CALL(cudaMemcpyFromSymbolAsync(
        static_cast<void *>(hNumToBeDeleted), dNumToBeDeleted, sizeof(int), 0,
        cudaMemcpyDeviceToHost, params.stream1));

    // Increment the path of each bubble
    KERNEL_LAUNCH(incrementPath, params, 0, params.stream1, params.bubbles);
    CUDA_CALL(cudaEventRecord(params.event1, params.stream1));

    // In correct kernel each block reduced a maximum value of the encountered
    // radii and expansions. Reduce those per block maximums to a global
    // maximum.
    void *pDMaxRadius = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&pDMaxRadius, dMaxRadius));
    params.cw.reduceNoCopy<double, double *, double *>(
        &cub::DeviceReduce::Max, params.bubbles.tempDoubles2 + numBlocks,
        static_cast<double *>(pDMaxRadius), numBlocks, params.stream2);
    CUDA_CALL(cudaMemcpyFromSymbolAsync(
        static_cast<void *>(hMaxRadius), dMaxRadius, sizeof(double), 0,
        cudaMemcpyDeviceToHost, params.stream2));

    void *pDMaxExpansion = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&pDMaxExpansion, dMaxExpansion));
    params.cw.reduceNoCopy<double, double *, double *>(
        &cub::DeviceReduce::Max, params.bubbles.tempDoubles2 + 2 * numBlocks,
        static_cast<double *>(pDMaxExpansion), numBlocks, params.stream2);
    CUDA_CALL(cudaMemcpyFromSymbolAsync(
        static_cast<void *>(hMaxExpansion), dMaxExpansion, sizeof(double), 0,
        cudaMemcpyDeviceToHost, params.stream2));

    // Record event after both reductions & copies are done
    CUDA_CALL(cudaEventRecord(params.event2, params.stream2));

    // Update values
    double *swapper = params.bubbles.dxdto;
    params.bubbles.dxdto = params.bubbles.dxdt;
    params.bubbles.dxdt = params.bubbles.dxdtp;
    params.bubbles.dxdtp = swapper;

    swapper = params.bubbles.dydto;
    params.bubbles.dydto = params.bubbles.dydt;
    params.bubbles.dydt = params.bubbles.dydtp;
    params.bubbles.dydtp = swapper;

    swapper = params.bubbles.dzdto;
    params.bubbles.dzdto = params.bubbles.dzdt;
    params.bubbles.dzdt = params.bubbles.dzdtp;
    params.bubbles.dzdtp = swapper;

    swapper = params.bubbles.drdto;
    params.bubbles.drdto = params.bubbles.drdt;
    params.bubbles.drdt = params.bubbles.drdtp;
    params.bubbles.drdtp = swapper;

    swapper = params.bubbles.x;
    params.bubbles.x = params.bubbles.xp;
    params.bubbles.xp = swapper;

    swapper = params.bubbles.y;
    params.bubbles.y = params.bubbles.yp;
    params.bubbles.yp = swapper;

    swapper = params.bubbles.z;
    params.bubbles.z = params.bubbles.zp;
    params.bubbles.zp = swapper;

    swapper = params.bubbles.r;
    params.bubbles.r = params.bubbles.rp;
    params.bubbles.rp = swapper;

    ++params.hostData.numIntegrationSteps;

    // As the total simulation time can reach very large numbers as the
    // simulation goes on it's better to keep track of the time as two separate
    // values. One large integer for the integer part and a double that is
    // <= 1.0 to which the potentially very small timeStep gets added. This
    // keeps the precision of the time relatively constant even when the
    // simulation has run a long time.
    params.hostData.timeFraction += params.hostData.timeStep;
    params.hostData.timeInteger += (uint64_t)params.hostData.timeFraction;
    params.hostData.timeFraction =
        params.hostData.timeFraction - (uint64_t)params.hostData.timeFraction;

    // Delete, if there are nonzero amount of bubbles with a radius
    // smaller than the minimum radius. See correct kernel for the
    // comparison & calculation.
    CUDA_CALL(cudaEventSynchronize(params.event1));
    if (*hNumToBeDeleted > 0) {
        deleteSmallBubbles(params, *hNumToBeDeleted);
    }

    // If the boundary of the bubble with maximum sum of movement & expansion
    // has moved more than half of the "skin radius", reorder bubbles.
    // See correct kernel, comparePair for details.
    CUDA_CALL(cudaEventSynchronize(params.event2));
    params.hostData.maxBubbleRadius = *hMaxRadius;
    if (*hMaxExpansion >= 0.5 * params.hostConstants.skinRadius) {
        updateCellsAndNeighbors(params);
    }

    bool continueSimulation =
        params.bubbles.count > params.hostData.minNumBubbles;
    continueSimulation &=
        (params.hostConstants.dimensionality == 3)
            ? params.hostData.maxBubbleRadius <
                  0.5 * params.hostConstants.interval.getMinComponent()
            : true;

    NVTX_RANGE_POP();
    return continueSimulation;
}
} // namespace

namespace // anonymous
{
using namespace cubble;
#if (USE_PROFILING == 1)
void startProfiling(bool start) {
    if (start) {
        CUDA_CALL(cudaProfilerStart());
    }
}

void stopProfiling(bool stop, bool &continueIntegration) {
    if (stop) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaProfilerStop());
        continueIntegration = false;
    }
}
#endif

double calculateTotalEnergy(Params &params) {
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.bubbles.tempDoubles);
    KERNEL_LAUNCH(potentialEnergy, params, 0, 0, params.bubbles, params.pairs);
    return params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.bubbles.tempDoubles,
        params.bubbles.count);
}

double calculateVolumeOfBubbles(Params &params) {
    KERNEL_LAUNCH(calculateVolumes, params, 0, 0, params.bubbles);
    return params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.bubbles.tempDoubles,
        params.bubbles.count);
}

double getSimulationBoxVolume(Params &params) {
    dvec temp = params.hostConstants.interval;
    return (params.hostConstants.dimensionality == 3) ? temp.x * temp.y * temp.z
                                                      : temp.x * temp.y;
}

void saveSnapshotToFile(Params &params) {
    // Should measure at some point how long it takes to save a snapshot
    // since there are many optimization possibilities here.
    calculateTotalEnergy(params);

    std::stringstream ss;
    ss << params.hostData.snapshotFilename << ".csv."
       << params.hostData.numSnapshots;
    std::ofstream file(ss.str().c_str(), std::ios::out);
    if (file.is_open()) {
        // Copy entire bubble struct to host memory
        uint64_t bytes = params.bubbles.getMemReq();
        std::vector<char> rawMem;
        rawMem.resize(bytes);
        void *memStart = static_cast<void *>(rawMem.data());
        // Async copy so host can sort pointers while copy is happening
        CUDA_CALL(cudaMemcpyAsync(memStart, params.memory, bytes,
                                  cudaMemcpyDeviceToHost, 0));

        // Get host pointer for each device pointer
        auto getHostPtr = [&params,
                           &memStart](auto devPtr) -> decltype(devPtr) {
            return static_cast<decltype(devPtr)>(memStart) +
                   (devPtr - static_cast<decltype(devPtr)>(params.memory));
        };
        double *x = getHostPtr(params.bubbles.x);
        double *y = getHostPtr(params.bubbles.y);
        double *z = getHostPtr(params.bubbles.z);
        double *r = getHostPtr(params.bubbles.r);
        double *vx = getHostPtr(params.bubbles.dxdt);
        double *vy = getHostPtr(params.bubbles.dydt);
        double *vz = getHostPtr(params.bubbles.dzdt);
        double *vr = getHostPtr(params.bubbles.drdt);
        double *path = getHostPtr(params.bubbles.path);
        double *error = getHostPtr(params.bubbles.error);
        double *energy = getHostPtr(params.bubbles.tempDoubles);
        int *index = getHostPtr(params.bubbles.index);

        // Starting to access the data, so need to sync to make sure all the
        // data is there
        CUDA_CALL(cudaDeviceSynchronize());

        if (params.hostData.numSnapshots == 0) {
            // If this is the first snapshot, store current positions in the
            // previous
            for (uint64_t i = 0; i < (uint64_t)params.bubbles.count; ++i) {
                const int ind = index[i];
                params.previousX[ind] = x[i];
                params.previousY[ind] = y[i];
                params.previousZ[ind] = z[i];
            }
        }

        file << "x,y,z,r,vx,vy,vz,vtot,vr,path,energy,displacement,"
                "error,index\n ";
        for (uint64_t i = 0; i < (uint64_t)params.bubbles.count; ++i) {
            const int ind = index[i];
            const double xi = x[i];
            const double yi = y[i];
            const double zi = z[i];
            const double vxi = vx[i];
            const double vyi = vy[i];
            const double vzi = vz[i];
            const double px = params.previousX[ind];
            const double py = params.previousY[ind];
            const double pz = params.previousZ[ind];

            double displX = abs(xi - px);
            displX = displX > 0.5 * params.hostConstants.interval.x
                         ? displX - params.hostConstants.interval.x
                         : displX;
            double displY = abs(yi - py);
            displY = displY > 0.5 * params.hostConstants.interval.y
                         ? displY - params.hostConstants.interval.y
                         : displY;
            double displZ = abs(zi - pz);
            displZ = displZ > 0.5 * params.hostConstants.interval.z
                         ? displZ - params.hostConstants.interval.z
                         : displZ;

            file << xi;
            file << ",";
            file << yi;
            file << ",";
            file << zi;
            file << ",";
            file << r[i];
            file << ",";
            file << vxi;
            file << ",";
            file << vyi;
            file << ",";
            file << vzi;
            file << ",";
            file << sqrt(vxi * vxi + vyi * vyi + vzi * vzi);
            file << ",";
            file << vr[i];
            file << ",";
            file << path[i];
            file << ",";
            file << energy[i];
            file << ",";
            file << sqrt(displX * displX + displY * displY + displZ * displZ);
            file << ",";
            file << error[i];
            file << ",";
            file << ind;
            file << "\n";

            params.previousX[ind] = xi;
            params.previousY[ind] = yi;
            params.previousZ[ind] = zi;
        }

        ++params.hostData.numSnapshots;
    }
}

void deinit(Params &params) {
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(params.deviceConstants)));
    CUDA_CALL(cudaFree(params.memory));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinnedMemory)));

    CUDA_CALL(cudaEventDestroy(params.event1));
    CUDA_CALL(cudaEventDestroy(params.event2));

    CUDA_CALL(cudaStreamDestroy(params.stream2));
    CUDA_CALL(cudaStreamDestroy(params.stream1));
}

void commonSetup(Params &params) {
    CUDA_ASSERT(cudaStreamCreate(&params.stream2));
    CUDA_ASSERT(cudaStreamCreate(&params.stream1));
    CUDA_CALL(cudaEventCreate(&params.event1));
    CUDA_CALL(cudaEventCreate(&params.event2));
    printRelevantInfoOfCurrentDevice();

    // Set device globals to zero
    double zero = 0.0;
    void *vz = reinterpret_cast<void *>(&zero);
    CUDA_CALL(cudaMemcpyToSymbol(dTotalArea, vz, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dTotalOverlapArea, vz, sizeof(double)));
    CUDA_CALL(
        cudaMemcpyToSymbol(dTotalOverlapAreaPerRadius, vz, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dTotalAreaPerRadius, vz, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dTotalVolumeNew, vz, sizeof(double)));

    bool falseB = false;
    vz = reinterpret_cast<void *>(&falseB);
    CUDA_CALL(cudaMemcpyToSymbol(dErrorEncountered, vz, sizeof(bool)));

    int zeroI = 0;
    vz = reinterpret_cast<void *>(&zeroI);
    CUDA_CALL(cudaMemcpyToSymbol(dNumPairs, vz, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(dNumPairsNew, vz, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(dNumToBeDeleted, vz, sizeof(int)));

    std::cout << "Reserving device memory to hold data." << std::endl;
    CUDA_CALL(
        cudaMallocHost(&params.pinnedMemory, sizeof(int) + 2 * sizeof(double)));

    // It seems to hold that in 3 dimensions the total number of
    // bubble pairs is 10x and in two dimensions 4x number of bubbles.
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avgNumNeighbors =
        (params.hostConstants.dimensionality == 3) ? 10 : 4;

    // Calculate the length of 'rows'. Will be divisible by 32, as that's the
    // warp size.
    params.bubbles.stride =
        params.bubbles.count +
        !!(params.bubbles.count % 32) * (32 - params.bubbles.count % 32);
    params.pairs.stride = avgNumNeighbors * params.bubbles.stride;

    uint64_t bytes = params.bubbles.getMemReq();
    bytes += params.pairs.getMemReq();

    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.memory), bytes));

    // Each named pointer is setup by these functions to point to
    // a different stride inside the continuous memory blob
    void *pairStart = params.bubbles.setupPointers(params.memory);
    pairStart = params.pairs.setupPointers(pairStart);

    params.previousX.resize(params.bubbles.stride);
    params.previousY.resize(params.bubbles.stride);
    params.previousZ.resize(params.bubbles.stride);

    const uint64_t megs = bytes / (1024 * 1024);
    const uint64_t kilos = (bytes - megs * 1024 * 1024) / 1024;
    bytes = (bytes - megs * 1024 * 1024 - kilos * 1024);
    std::cout << "Total device memory allocated: " << megs << "m " << kilos
              << "k " << bytes << "b" << std::endl;
}

void generateStartingData(Params &params, ivec bubblesPerDim, double stdDevRad,
                          int rngSeed) {
    std::cout << "Starting to generate data for bubbles." << std::endl;
    const double avgRad = params.hostData.avgRad;

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    if (params.hostConstants.dimensionality == 3)
        CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.z,
                                                params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.x,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.y,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.rp,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateNormalDouble(
        generator, params.bubbles.r, params.bubbles.count, avgRad, stdDevRad));
    CURAND_CALL(curandDestroyGenerator(generator));

    KERNEL_LAUNCH(assignDataToBubbles, params, 0, 0, bubblesPerDim, avgRad,
                  params.bubbles);

    params.hostConstants.averageSurfaceAreaIn =
        params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Sum, params.bubbles.rp, params.bubbles.count);

    params.hostData.maxBubbleRadius =
        params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Max, params.bubbles.r, params.bubbles.count);

    std::cout << "Updating neighbor lists." << std::endl;
    updateCellsAndNeighbors(params);

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton predictor-corrector method
    KERNEL_LAUNCH(
        resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
        params.bubbles.dxdto, params.bubbles.dydto, params.bubbles.dzdto,
        params.bubbles.drdto, params.bubbles.dxdtp, params.bubbles.dydtp,
        params.bubbles.dzdtp, params.bubbles.drdtp, params.bubbles.path);

    std::cout << "Calculating some initial values as a part of setup."
              << std::endl;

    // After updateCellsAndNeighbors x, y, z, r are correct,
    // but all predicted are trash. pairVelocity always uses
    // predicted values, so copy currents to predicteds
    uint64_t bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.xp),
                              static_cast<void *>(params.bubbles.x), bytes,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.yp),
                              static_cast<void *>(params.bubbles.y), bytes,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.zp),
                              static_cast<void *>(params.bubbles.z), bytes,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDeviceToDevice, 0));

    KERNEL_LAUNCH(pairVelocity, params, 0, 0, params.bubbles, params.pairs);

    KERNEL_LAUNCH(euler, params, 0, 0, params.hostData.timeStep,
                  params.bubbles);

    // pairVelocity calculates to predicteds by accumulating values
    // using atomicAdd. They would have to be reset to zero after every
    // integration, but olds were set to zero above, so we can just swap.
    double *swapper = params.bubbles.dxdto;
    params.bubbles.dxdto = params.bubbles.dxdtp;
    params.bubbles.dxdtp = swapper;

    swapper = params.bubbles.dydto;
    params.bubbles.dydto = params.bubbles.dydtp;
    params.bubbles.dydtp = swapper;

    swapper = params.bubbles.dzdto;
    params.bubbles.dzdto = params.bubbles.dzdtp;
    params.bubbles.dzdtp = swapper;

    KERNEL_LAUNCH(pairVelocity, params, 0, 0, params.bubbles, params.pairs);

    // The whole point of this part was to get integrated values into
    // dxdto & y & z, so swap again so that predicteds are in olds.
    swapper = params.bubbles.dxdto;
    params.bubbles.dxdto = params.bubbles.dxdtp;
    params.bubbles.dxdtp = swapper;

    swapper = params.bubbles.dydto;
    params.bubbles.dydto = params.bubbles.dydtp;
    params.bubbles.dydtp = swapper;

    swapper = params.bubbles.dzdto;
    params.bubbles.dzdto = params.bubbles.dzdtp;
    params.bubbles.dzdtp = swapper;
}

void initializeFromJson(const char *inputFileName, Params &params) {
    std::cout << "\n=====\nSetup\n=====\n"
              << "Reading inputs from file \"" << inputFileName << "\""
              << std::endl;

    nlohmann::json inputJson;
    std::fstream file(inputFileName, std::ios::in);
    if (file.is_open()) {
        file >> inputJson;
    } else {
        throw std::runtime_error("Couldn't open input file!");
    }

    auto constants = inputJson["constants"];
    auto bubbles = inputJson["bubbles"];
    auto box = inputJson["box"];
    auto wall = box["wall"];
    auto flow = inputJson["flow"];

    const int stabilizationSteps = inputJson["stabilization"]["steps"];

    params.hostData.avgRad = bubbles["radius"]["mean"];
    params.hostData.minNumBubbles = bubbles["numEnd"];
    params.hostConstants.skinRadius *= params.hostData.avgRad;

    const double mu = constants["mu"]["value"];
    const double phi = constants["phi"]["value"];
    params.hostConstants.minRad = 0.1 * params.hostData.avgRad;
    params.hostConstants.fZeroPerMuZero =
        (float)constants["sigma"]["value"] * params.hostData.avgRad / mu;
    params.hostConstants.kParameter = constants["K"]["value"];
    params.hostConstants.kappa = constants["kappa"]["value"];
    params.hostData.timeScalingFactor =
        params.hostConstants.kParameter /
        (params.hostData.avgRad * params.hostData.avgRad);

    params.hostData.addFlow = flow["impose"];
    params.hostConstants.flowLbb = flow["lbb"];
    params.hostConstants.flowTfr = flow["tfr"];
    params.hostConstants.flowVel = flow["velocity"];
    params.hostConstants.flowVel *= params.hostConstants.fZeroPerMuZero;

    params.hostData.errorTolerance = inputJson["errorTolerance"]["value"];
    params.hostData.snapshotFrequency = inputJson["snapShot"]["frequency"];
    params.hostData.snapshotFilename = inputJson["snapShot"]["filename"];

    params.hostConstants.wallDragStrength = wall["drag"];
    params.hostConstants.xWall = wall["x"];
    params.hostConstants.yWall = wall["y"];
    params.hostConstants.zWall = wall["z"];
    params.hostConstants.dimensionality = box["dimensionality"];

    // First calculate the size of the box and the starting number of bubbles
    dvec relDim = box["relativeDimensions"];
    relDim = relDim / relDim.x;
    const float d = 2 * params.hostData.avgRad;
    float x = (float)bubbles["numStart"] * d * d / relDim.y;
    ivec bubblesPerDim = ivec(0, 0, 0);

    if (params.hostConstants.dimensionality == 3) {
        x = x * d / relDim.z;
        x = std::cbrt(x);
        relDim = relDim * x;
        bubblesPerDim = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d),
                             std::ceil(relDim.z / d));
        params.bubbles.count =
            bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
    } else {
        x = std::sqrt(x);
        relDim = relDim * x;
        bubblesPerDim =
            ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
        params.bubbles.count = bubblesPerDim.x * bubblesPerDim.y;
    }

    params.hostConstants.tfr =
        d * bubblesPerDim.asType<double>() + params.hostConstants.lbb;
    params.hostConstants.interval =
        params.hostConstants.tfr - params.hostConstants.lbb;

    // Allocate and copy constants to GPU
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceConstants),
                           sizeof(Constants)));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceConstants),
                         static_cast<void *>(&params.hostConstants),
                         sizeof(Constants), cudaMemcpyHostToDevice));
    // Copy to global pointer
    CUDA_CALL(cudaMemcpyToSymbol(dConstants,
                                 static_cast<void *>(&params.deviceConstants),
                                 sizeof(Constants *)));

    // Reserve memory etc.
    commonSetup(params);
    generateStartingData(params, bubblesPerDim, bubbles["radius"]["std"],
                         inputJson["rngSeed"]["value"]);

    std::cout << "Simulation starting parameters:\n";
    params.hostConstants.print();
    params.hostData.print();
    params.bubbles.print();
    params.pairs.print();

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params, stabilizationSteps);

    const double bubbleVolume = calculateVolumeOfBubbles(params);
    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << phi << "\nScaling the simulation box."
              << std::endl;

    KERNEL_LAUNCH(transformPositions, params, 0, 0, true, params.bubbles);

    relDim = box["relativeDimensions"];
    double t = bubbleVolume / (phi * relDim.x * relDim.y);
    if (params.hostConstants.dimensionality == 3) {
        t /= relDim.z;
        t = std::cbrt(t);
    } else {
        t = std::sqrt(t);
        relDim.z = 0.0;
    }

    params.hostConstants.tfr = dvec(t, t, t) * relDim;
    params.hostConstants.interval =
        params.hostConstants.tfr - params.hostConstants.lbb;
    params.hostConstants.flowTfr =
        params.hostConstants.interval * params.hostConstants.flowTfr +
        params.hostConstants.lbb;
    params.hostConstants.flowLbb =
        params.hostConstants.interval * params.hostConstants.flowLbb +
        params.hostConstants.lbb;

    double mult = phi * getSimulationBoxVolume(params) / CUBBLE_PI;
    if (params.hostConstants.dimensionality == 3) {
        mult = std::cbrt(0.75 * mult);
    } else {
        mult = std::sqrt(mult);
    }
    params.hostConstants.bubbleVolumeMultiplier = mult;

    // Copy the updated constants to GPU
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceConstants),
                         static_cast<void *>(&params.hostConstants),
                         sizeof(Constants), cudaMemcpyHostToDevice));

    KERNEL_LAUNCH(transformPositions, params, 0, 0, false, params.bubbles);

    updateCellsAndNeighbors(params);
    // After updateCellsAndNeighbors r is correct,
    // but rp is trash. pairVelocity always uses
    // predicted values, so copy r to rp
    uint64_t bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDeviceToDevice, 0));

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params, stabilizationSteps);

    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << phi
              << "\n\n=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    std::cout << std::setw(10) << std::left << "#steps" << std::setw(12)
              << std::left << "dE" << std::setw(15) << std::left << "e1"
              << std::setw(15) << std::left << "e2" << std::setw(5) << std::left
              << "#searches" << std::endl;

    while (true) {
        double time = stabilize(params, stabilizationSteps);
        double deltaEnergy =
            std::abs(1.0 - params.hostData.energy1 / params.hostData.energy2) /
            time;

        if (deltaEnergy < inputJson["stabilization"]["maxDeltaEnergy"]) {
            std::cout << "Final delta energy " << deltaEnergy << " after "
                      << (numSteps + 1) * stabilizationSteps << " steps."
                      << "\nEnergy before: " << params.hostData.energy1
                      << ", energy after: " << params.hostData.energy2
                      << ", time: " << time * params.hostData.timeScalingFactor
                      << std::endl;
            break;
        } else if (numSteps > failsafe) {
            std::cout << "Over " << failsafe * stabilizationSteps
                      << " steps taken and required delta energy not reached."
                      << " Check parameters." << std::endl;
            break;
        } else {
            std::cout << std::setw(10) << std::left
                      << (numSteps + 1) * stabilizationSteps << std::setw(12)
                      << std::left << std::setprecision(5) << std::scientific
                      << deltaEnergy << std::setw(15) << std::left
                      << std::setprecision(5) << std::fixed
                      << params.hostData.energy1 << std::setw(15) << std::left
                      << std::setprecision(5) << std::fixed
                      << params.hostData.energy2 << std::setw(5) << std::left
                      << params.hostData.numNeighborsSearched << std::endl;
            params.hostData.numNeighborsSearched = 0;
        }

        ++numSteps;
    }

    // TODO: Set starting positions
    // Avoiding batched memset, because the pointers might not be in order

    // Reset wrap counts to 0
    // Again avoiding batched memset, because the pointers might not be in order
    bytes = sizeof(int) * params.bubbles.stride;
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountX, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountY, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountZ, 0, bytes));

    // Reset errors since integration starts after this
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.bubbles.error);

    params.hostData.energy1 = calculateTotalEnergy(params);
    params.hostData.timeInteger = 0;
    params.hostData.timeFraction = 0.0;
    params.hostData.timesPrinted = 1;
    params.hostData.numIntegrationSteps = 0;
}

}; // namespace

namespace cubble {
void run(std::string &&inputFileName) {
    Params params;
    initializeFromJson(inputFileName.c_str(), params);
    if (params.hostData.snapshotFrequency > 0.0)
        saveSnapshotToFile(params);

    std::cout << "\n==========\nIntegration\n==========" << std::endl;

    std::cout << std::setw(10) << std::left << "T" << std::setw(10) << std::left
              << "phi" << std::setw(10) << std::left << "R" << std::setw(10)
              << std::left << "#b" << std::setw(10) << std::left << "#pairs"
              << std::setw(10) << std::left << "#steps" << std::setw(10)
              << std::left << "#searches" << std::setw(10) << std::left
              << "min ts" << std::setw(10) << std::left << "max ts"
              << std::setw(10) << std::left << "avg ts" << std::endl;

    bool continueIntegration = true;
    double minTimestep = 9999999.9;
    double maxTimestep = -1.0;
    double avgTimestep = 0.0;
    bool resetErrors = false;

    // This is the simulation loop, which runs until (at least) one
    // end condition is met
    while (continueIntegration) {
        // Perform one integration step
        continueIntegration = integrate(params);

        // When profiling, we don't want to run the entire simulation until the
        // end, but rather just enough simulation steps to get a representative
        // view of the entire simulation
        CUDA_PROFILER_START(params.hostData.numIntegrationSteps == 2000);
        CUDA_PROFILER_STOP(params.hostData.numIntegrationSteps == 12000,
                           continueIntegration);

        // Track timestep
        minTimestep = params.hostData.timeStep < minTimestep
                          ? params.hostData.timeStep
                          : minTimestep;
        maxTimestep = params.hostData.timeStep > maxTimestep
                          ? params.hostData.timeStep
                          : maxTimestep;
        avgTimestep += params.hostData.timeStep;

        // Here we compare potentially very large integers (> 10e6) to each
        // other and small doubles (<= 1.0) to each other to preserve precision.
        const double nextPrintTime =
            params.hostData.timesPrinted / params.hostData.timeScalingFactor;
        const uint64_t nextPrintTimeInteger = (uint64_t)nextPrintTime;
        const double nextPrintTimeFraction =
            nextPrintTime - nextPrintTimeInteger;

        // Print stuff to stdout at the earliest possible moment
        // when simulation time is larger than scaled time
        const bool print =
            params.hostData.timeInteger > nextPrintTimeInteger ||
            (params.hostData.timeInteger == nextPrintTimeInteger &&
             params.hostData.timeFraction >= nextPrintTimeFraction);
        if (print) {
            // Scale the tracked timesteps
            minTimestep *= params.hostData.timeScalingFactor;
            maxTimestep *= params.hostData.timeScalingFactor;
            avgTimestep *= params.hostData.timeScalingFactor;

            // Define lambda for calculating averages of some values
            auto getAvg = [&params](double *p, Bubbles &bubbles) -> double {
                return params.cw.reduce<double, double *, double *>(
                           &cub::DeviceReduce::Sum, p, bubbles.count) /
                       bubbles.count;
            };

            params.hostData.energy2 = calculateTotalEnergy(params);
            const double dE =
                (params.hostData.energy2 - params.hostData.energy1) /
                params.hostData.energy2;
            const double relRad = getAvg(params.bubbles.r, params.bubbles) /
                                  params.hostData.avgRad;

            // Add values to data stream
            std::ofstream resultFile("results.dat", std::ios_base::app);
            if (resultFile.is_open()) {
                const double vx = getAvg(params.bubbles.dxdt, params.bubbles);
                const double vy = getAvg(params.bubbles.dydt, params.bubbles);
                const double vz = getAvg(params.bubbles.dzdt, params.bubbles);
                const double vr = getAvg(params.bubbles.drdt, params.bubbles);

                resultFile << params.hostData.timesPrinted << " " << relRad
                           << " " << params.bubbles.count << " "
                           << getAvg(params.bubbles.path, params.bubbles) << " "
                           << params.hostData.energy2 << " " << dE << " " << vx
                           << " " << vy << " " << vz << " "
                           << sqrt(vx * vx + vy * vy + vz * vz) << " " << vr
                           << "\n";
            } else {
                std::cout << "Couldn't open file stream to append results to!"
                          << std::endl;
            }

            const double phi = calculateVolumeOfBubbles(params) /
                               getSimulationBoxVolume(params);

            // Print some values
            std::cout << std::setw(10) << std::left
                      << params.hostData.timesPrinted << std::setw(10)
                      << std::left << std::setprecision(6) << std::fixed << phi
                      << std::setw(10) << std::left << std::setprecision(6)
                      << std::fixed << relRad << std::setw(10) << std::left
                      << params.bubbles.count << std::setw(10) << std::left
                      << params.pairs.count << std::setw(10) << std::left
                      << params.hostData.numStepsInTimeStep << std::setw(10)
                      << std::left << params.hostData.numNeighborsSearched
                      << std::setw(10) << std::left << minTimestep
                      << std::setw(10) << std::left << maxTimestep
                      << std::setw(10) << std::left
                      << avgTimestep / params.hostData.numStepsInTimeStep
                      << std::endl;

            ++params.hostData.timesPrinted;
            params.hostData.numStepsInTimeStep = 0;
            params.hostData.energy1 = params.hostData.energy2;
            params.hostData.numNeighborsSearched = 0;
            minTimestep = 9999999.9;
            maxTimestep = -1.0;
            avgTimestep = 0.0;
            resetErrors = true;
        }

        // Save snapshot
        if (params.hostData.snapshotFrequency > 0.0) {
            const double nextSnapshotTime = params.hostData.numSnapshots /
                                            (params.hostData.snapshotFrequency *
                                             params.hostData.timeScalingFactor);
            const uint64_t nextSnapshotTimeInteger = (uint64_t)nextSnapshotTime;
            const double nextSnapshotTimeFraction =
                nextSnapshotTime - nextSnapshotTimeInteger;

            const bool saveSnapshot =
                params.hostData.timeInteger > nextSnapshotTimeInteger ||
                (params.hostData.timeInteger == nextSnapshotTimeInteger &&
                 params.hostData.timeFraction >= nextSnapshotTimeFraction);

            if (saveSnapshot) {
                saveSnapshotToFile(params);
            }
        }

        if (resetErrors) {
            KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count,
                          false, params.bubbles.error);
            resetErrors = false;
        }

        ++params.hostData.numStepsInTimeStep;
    }

    if (params.hostData.snapshotFrequency > 0.0) {
        saveSnapshotToFile(params);
    }

    deinit(params);
}
} // namespace cubble
