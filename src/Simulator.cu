#include "DataDefinitions.h"
#include "Kernels.cuh"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <fstream>
#include <functional>
#include <nvToolsExt.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

namespace // anonymous
{
using namespace cubble;
double totalEnergy(Params &params);

void searchNeighbors(Params &params) {
    nvtxRangePush("Neighbors");
    params.hostData.numNeighborsSearched++;

    KERNEL_LAUNCH(wrapOverPeriodicBoundaries, params, 0, params.stream1,
                  params.bubbles);

    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.pairs.i), 0,
                              params.pairs.getMemReq(), params.stream2));

    // Minimum size of cell is twice the sum of the skin and max bubble radius
    ivec cellDim = (params.hostConstants.interval /
                    (2 * (params.hostData.maxBubbleRadius +
                          params.hostConstants.skinRadius)))
                       .floor();
    cellDim.z = cellDim.z > 0 ? cellDim.z : 1;
    const int numCells = cellDim.x * cellDim.y * cellDim.z;

    // Note that these pointers alias memory, that is used by this function.
    // Don't fiddle with these, unless you know what you're doing.
    int *cellOffsets = params.pairs.i;
    int *cellSizes = cellOffsets + numCells;
    int *cellIndices = cellSizes + numCells;
    int *bubbleIndices = cellIndices + params.bubbles.stride;
    int *histogram = bubbleIndices + params.bubbles.stride;
    void *cubPtr = static_cast<void *>(params.pairs.j);
    uint64_t maxCubMem = params.pairs.getMemReq() / 2;

    KERNEL_LAUNCH(cellByPosition, params, 0, 0, cellIndices, cellSizes, cellDim,
                  params.bubbles);

    CUB_LAUNCH(&cub::DeviceScan::InclusiveSum, cubPtr, maxCubMem, cellSizes,
               cellOffsets, numCells, (cudaStream_t)0, false);

    KERNEL_LAUNCH(indexByCell, params, 0, 0, cellIndices, cellOffsets,
                  bubbleIndices, params.bubbles.count);

    {
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
    }

    int zero = 0;
    CUDA_CALL(
        cudaMemcpyToSymbol(dNumPairs, static_cast<void *>(&zero), sizeof(int)));

    int numCellsToSearch = 5;
    if (params.hostConstants.dimensionality == 3) {
        numCellsToSearch = 14;
    }

    KERNEL_LAUNCH(neighborSearch, params, 0, 0, numCells, numCellsToSearch,
                  cellDim, cellOffsets, cellSizes, histogram, params.tempPair1,
                  params.tempPair2, params.bubbles, params.pairs);

    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&params.pairs.count),
                                   dNumPairs, sizeof(int)));

    CUB_LAUNCH(&cub::DeviceScan::InclusiveSum, cubPtr, maxCubMem, histogram,
               params.bubbles.numNeighbors, params.bubbles.count,
               (cudaStream_t)0, false);

    KERNEL_LAUNCH(sortPairs, params, 0, 0, params.bubbles, params.pairs,
                  params.tempPair1, params.tempPair2);

    CUDA_CALL(cudaMemset(static_cast<void *>(params.bubbles.numNeighbors), 0,
                         params.bubbles.count * sizeof(int)));

    KERNEL_LAUNCH(countNumNeighbors, params, 0, 0, params.bubbles,
                  params.pairs);
    nvtxRangePop();
}

void removeBubbles(Params &params, int numToBeDeleted) {
    nvtxRangePush("Removal");
    KERNEL_LAUNCH(swapDataCountPairs, params, 0, 0, params.bubbles,
                  params.pairs, params.tempI);

    KERNEL_LAUNCH(addVolumeFixPairs, params, 0, 0, params.bubbles, params.pairs,
                  params.tempI);

    params.bubbles.count -= numToBeDeleted;
    const int numBlocks =
        std::min(1024, (int)std::ceil(params.bubbles.count / 128.0));
    params.blockGrid = dim3(numBlocks, 1, 1);
    nvtxRangePop();
}

double stabilize(Params &params, int numStepsToRelax) {
    nvtxRangePush("Stabilization");
    // This function integrates only the positions of the bubbles.
    // Gas exchange is not used. This is used for equilibrating the foam.
    params.hostData.energy1 = totalEnergy(params);

    double elapsedTime = 0.0;
    double error = 100000;
    const int numBlocks = params.blockGrid.x;
    bool errorTooLarge = true;
    double &ts = params.hostData.timeStep;

    void *cubPtr = params.tempPair2;
    uint64_t maxCubMem = params.pairs.getMemReq() / 2;
    void *cubOutput = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&cubOutput, dMaxRadius));

    nvtxRangePush("For-loop");
    for (int i = 0; i < numStepsToRelax; ++i) {
        do {
            nvtxRangePush("Do-loop");
            KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count,
                          true, params.bubbles.dxdtp, params.bubbles.dydtp,
                          params.bubbles.dzdtp);

            KERNEL_LAUNCH(predict, params, 0, 0, ts, false, params.bubbles);

            KERNEL_LAUNCH(pairVelocity, params, 0, 0, params.bubbles,
                          params.pairs);

            if (params.hostConstants.xWall || params.hostConstants.yWall ||
                params.hostConstants.zWall) {
                KERNEL_LAUNCH(wallVelocity, params, 0, 0, params.bubbles);
            }

            KERNEL_LAUNCH(correct, params, 0, 0, ts, false, params.bubbles,
                          params.tempD2, params.tempI);

            // Reduce error
            CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, maxCubMem,
                       params.tempD2, static_cast<double *>(cubOutput),
                       numBlocks, (cudaStream_t)0, false);
            CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&error),
                                           dMaxRadius, sizeof(double)));

            errorTooLarge = error > params.hostData.errorTolerance;
            bool increaseTs = error < 0.45 * params.hostData.errorTolerance;
            if (errorTooLarge) {
                ts *= 0.37;
            } else if (increaseTs) {
                ts *= 1.269;
            }
            nvtxRangePop();
        } while (errorTooLarge);

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

        elapsedTime += ts;

        // Reduce block maximum expansions to a global maximum
        double maxExpansion = 0.0;
        CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, maxCubMem,
                   params.tempD2 + 2 * numBlocks,
                   static_cast<double *>(cubOutput), numBlocks, (cudaStream_t)0,
                   false);
        CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&maxExpansion),
                                       dMaxRadius, sizeof(double)));

        if (maxExpansion >= 0.5 * params.hostConstants.skinRadius) {
            searchNeighbors(params);
            // After searchNeighbors r is correct,
            // but rp is trash. pairVelocity always uses
            // predicted values, so copy r to rp
            uint64_t bytes = params.bubbles.stride * sizeof(double);
            CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                                      static_cast<void *>(params.bubbles.r),
                                      bytes, cudaMemcpyDeviceToDevice, 0));
        }
    }

    nvtxRangePop();
    params.hostData.energy2 = totalEnergy(params);
    nvtxRangePop();

    return elapsedTime;
}

void integrate(Params &params) {
    nvtxRangePush("Intergration");
    uint32_t numLoopsDone = 0;
    const int numBlocks = params.blockGrid.x;
    double &ts = params.hostData.timeStep;
    bool errorTooLarge = true;

    double *hMaxRadius = static_cast<double *>(params.pinnedMemory);
    double *hMaxExpansion = hMaxRadius + 1;
    double *hMaxError = hMaxExpansion + 1;
    int *hNumToBeDeleted = reinterpret_cast<int *>(hMaxError + 1);

    void *cubPtr = params.tempPair2;
    uint64_t maxCubMem = params.pairs.getMemReq() / 2;

    if (params.hostData.addFlow) {
        // Average neighbor velocity is calculated from velocities of previous
        // step.
        KERNEL_LAUNCH(resetArrays, params, 0, params.stream2, 0.0,
                      params.bubbles.count, false, params.bubbles.flowVx,
                      params.bubbles.flowVy, params.bubbles.flowVz);
        KERNEL_LAUNCH(averageNeighborVelocity, params, 0, params.stream2,
                      params.bubbles, params.pairs);
    }

    do {
        nvtxRangePush("Do-loop");
        // Stream1
        {
            KERNEL_LAUNCH(initGlobals, params, 0, params.stream1);
            CUDA_CALL(cudaEventRecord(params.event1, params.stream1));
            KERNEL_LAUNCH(resetArrays, params, 0, params.stream1, 0.0,
                          params.bubbles.count, false, params.bubbles.drdtp,
                          params.tempD1);
            KERNEL_LAUNCH(predict, params, 0, params.stream1, ts, true,
                          params.bubbles);
            KERNEL_LAUNCH(pairwiseGasExchange, params, 0, params.stream1,
                          params.bubbles, params.pairs, params.tempD1);
            KERNEL_LAUNCH(mediatedGasExchange, params, 0, params.stream1,
                          params.bubbles, params.tempD1);
        }

        // Stream2
        {
            KERNEL_LAUNCH(resetArrays, params, 0, params.stream2, 0.0,
                          params.bubbles.count, false, params.bubbles.dxdtp,
                          params.bubbles.dydtp, params.bubbles.dzdtp,
                          params.tempD2);
            CUDA_CALL(cudaStreamWaitEvent(params.stream2, params.event1, 0));
            KERNEL_LAUNCH(predict, params, 0, params.stream2, ts, false,
                          params.bubbles);
            KERNEL_LAUNCH(pairVelocity, params, 0, params.stream2,
                          params.bubbles, params.pairs);

            if (params.hostData.addFlow) {
                KERNEL_LAUNCH(imposedFlowVelocity, params, 0, params.stream2,
                              params.bubbles);
            }

            if (params.hostConstants.xWall || params.hostConstants.yWall ||
                params.hostConstants.zWall) {
                KERNEL_LAUNCH(wallVelocity, params, 0, params.stream2,
                              params.bubbles);
            }
        }

        // Correction in default stream (implicit synchronization with streams)
        KERNEL_LAUNCH(correct, params, 0, 0, ts, true, params.bubbles,
                      params.tempD2, params.tempI);

        // Stream1
        {
            // Reduce error
            void *pDMaxError = nullptr;
            CUDA_CALL(cudaGetSymbolAddress(&pDMaxError, dMaxError));
            CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, maxCubMem,
                       params.tempD2, static_cast<double *>(pDMaxError),
                       numBlocks, params.stream1, false);
            CUDA_CALL(cudaMemcpyFromSymbolAsync(
                static_cast<void *>(hMaxError), dMaxError, sizeof(double), 0,
                cudaMemcpyDeviceToHost, params.stream1));
            CUDA_CALL(cudaEventRecord(params.event1, params.stream1));

            // Reduce expansion
            void *pDMaxExpansion = nullptr;
            CUDA_CALL(cudaGetSymbolAddress(&pDMaxExpansion, dMaxExpansion));
            CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, maxCubMem,
                       params.tempD2 + 2 * numBlocks,
                       static_cast<double *>(pDMaxExpansion), numBlocks,
                       params.stream1, false);
            CUDA_CALL(cudaMemcpyFromSymbolAsync(
                static_cast<void *>(hMaxExpansion), dMaxExpansion,
                sizeof(double), 0, cudaMemcpyDeviceToHost, params.stream1));
        }

        // Stream2
        {
            // Copy numToBeDeleted
            CUDA_CALL(cudaMemcpyFromSymbolAsync(
                static_cast<void *>(hNumToBeDeleted), dNumToBeDeleted,
                sizeof(int), 0, cudaMemcpyDeviceToHost, params.stream2));

            // Reduce radius
            void *pDMaxRadius = nullptr;
            CUDA_CALL(cudaGetSymbolAddress(&pDMaxRadius, dMaxRadius));
            CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, maxCubMem,
                       params.tempD2 + numBlocks,
                       static_cast<double *>(pDMaxRadius), numBlocks,
                       params.stream2, false);
            CUDA_CALL(cudaMemcpyFromSymbolAsync(
                static_cast<void *>(hMaxRadius), dMaxRadius, sizeof(double), 0,
                cudaMemcpyDeviceToHost, params.stream2));

            CUDA_CALL(cudaEventRecord(params.event2, params.stream2));
        }

        // Wait until the copy of maximum error is done
        CUDA_CALL(cudaEventSynchronize(params.event1));

        errorTooLarge = *hMaxError > params.hostData.errorTolerance;
        bool increaseTs = *hMaxError < 0.45 * params.hostData.errorTolerance;
        if (errorTooLarge) {
            ts *= 0.37;
        } else if (increaseTs) {
            ts *= 1.269;
        }

        ++numLoopsDone;
        nvtxRangePop();
    } while (errorTooLarge);

    // Increment the path of each bubble
    KERNEL_LAUNCH(incrementPath, params, 0, params.stream1, params.bubbles);
    CUDA_CALL(cudaEventRecord(params.event1, params.stream1));

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
    params.hostData.timeFraction += ts;
    params.hostData.timeInteger += (uint64_t)params.hostData.timeFraction;
    params.hostData.timeFraction =
        params.hostData.timeFraction - (uint64_t)params.hostData.timeFraction;

    CUDA_CALL(cudaEventSynchronize(params.event1));
    CUDA_CALL(cudaEventSynchronize(params.event2));
    params.hostData.maxBubbleRadius = *hMaxRadius;

    // Delete, if there are nonzero amount of bubbles with a radius
    // smaller than the minimum radius. See correct kernel for the
    // comparison & calculation.
    if (*hNumToBeDeleted > 0) {
        removeBubbles(params, *hNumToBeDeleted);
    }

    // If the boundary of the bubble with maximum sum of movement & expansion
    // has moved more than half of the "skin radius", reorder bubbles.
    // See correct kernel, comparePair for details.
    if (*hMaxExpansion >= 0.5 * params.hostConstants.skinRadius) {
        searchNeighbors(params);
    }

    nvtxRangePop();
}
} // namespace

namespace // anonymous
{
using namespace cubble;
double totalEnergy(Params &params) {
    nvtxRangePush("Energy");
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.tempD1);
    KERNEL_LAUNCH(potentialEnergy, params, 0, 0, params.bubbles, params.pairs,
                  params.tempD1);

    void *cubPtr = static_cast<void *>(params.tempPair2);
    double total = 0.0;
    void *cubOutput = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&cubOutput, dMaxRadius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cubPtr, params.pairs.getMemReq() / 2,
               params.tempD1, static_cast<double *>(cubOutput),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total), dMaxRadius,
                                   sizeof(double)));
    nvtxRangePop();

    return total;
}

double totalVolume(Params &params) {
    nvtxRangePush("Volume");
    KERNEL_LAUNCH(calculateVolumes, params, 0, 0, params.bubbles,
                  params.tempD1);

    void *cubPtr = static_cast<void *>(params.tempPair2);
    double total = 0.0;
    void *cubOutput = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&cubOutput, dMaxRadius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cubPtr, params.pairs.getMemReq() / 2,
               params.tempD1, static_cast<double *>(cubOutput),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total), dMaxRadius,
                                   sizeof(double)));
    nvtxRangePop();

    return total;
}

double boxVolume(Params &params) {
    dvec temp = params.hostConstants.interval;
    return (params.hostConstants.dimensionality == 3) ? temp.x * temp.y * temp.z
                                                      : temp.x * temp.y;
}

void saveSnapshot(Params &params) {
    // Calculate energies of bubbles to tempD1, but don't reduce.
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.tempD1);
    KERNEL_LAUNCH(potentialEnergy, params, 0, 0, params.bubbles, params.pairs,
                  params.tempD1);

    // Make sure the thread is not working
    if (params.ioThread.joinable()) {
        params.ioThread.join();
    }

    // Copy all device memory to host.
    void *memStart = static_cast<void *>(params.hostMemory.data());
    CUDA_CALL(
        cudaMemcpyAsync(memStart, params.memory,
                        params.hostMemory.size() * sizeof(params.hostMemory[0]),
                        cudaMemcpyDeviceToHost, 0));
    CUDA_CALL(cudaEventRecord(params.snapshotParams.event, 0));

    // This lambda helps calculate the host address for each pointer from the
    // device address.
    auto getHostPtr = [&params, &memStart](auto devPtr) -> decltype(devPtr) {
        return static_cast<decltype(devPtr)>(memStart) +
               (devPtr - static_cast<decltype(devPtr)>(params.memory));
    };

    // Get the host pointers from the device pointers
    params.snapshotParams.x = getHostPtr(params.bubbles.x);
    params.snapshotParams.y = getHostPtr(params.bubbles.y);
    params.snapshotParams.z = getHostPtr(params.bubbles.z);
    params.snapshotParams.r = getHostPtr(params.bubbles.r);
    params.snapshotParams.vx = getHostPtr(params.bubbles.dxdt);
    params.snapshotParams.vy = getHostPtr(params.bubbles.dydt);
    params.snapshotParams.vz = getHostPtr(params.bubbles.dzdt);
    params.snapshotParams.vr = getHostPtr(params.bubbles.drdt);
    params.snapshotParams.path = getHostPtr(params.bubbles.path);
    params.snapshotParams.error = getHostPtr(params.bubbles.error);
    params.snapshotParams.energy = getHostPtr(params.tempD1);
    params.snapshotParams.index = getHostPtr(params.bubbles.index);
    params.snapshotParams.wrapCountX = getHostPtr(params.bubbles.wrapCountX);
    params.snapshotParams.wrapCountY = getHostPtr(params.bubbles.wrapCountY);
    params.snapshotParams.wrapCountZ = getHostPtr(params.bubbles.wrapCountZ);

    params.snapshotParams.count = params.bubbles.count;

    // TODO: add distance from start
    auto writeSnapshot = [](const SnapshotParams &snapshotParams,
                            uint32_t snapshotNum, double *xPrev, double *yPrev,
                            double *zPrev) {
        std::stringstream ss;
        ss << snapshotParams.name << ".csv." << snapshotNum;
        std::ofstream file(ss.str().c_str(), std::ios::out);
        if (file.is_open()) {
            // Wait for the copy initiated by the main thread to be complete.
            CUDA_CALL(cudaEventSynchronize(snapshotParams.event));
            file << "x,y,z,r,vx,vy,vz,vtot,vr,path,energy,displacement,"
                    "error,index\n";
            for (uint64_t i = 0; i < snapshotParams.count; ++i) {
                const int ind = snapshotParams.index[i];
                const double xi = snapshotParams.x[i];
                const double yi = snapshotParams.y[i];
                const double zi = snapshotParams.z[i];
                const double vxi = snapshotParams.vx[i];
                const double vyi = snapshotParams.vy[i];
                const double vzi = snapshotParams.vz[i];
                const double px = xPrev[ind];
                const double py = yPrev[ind];
                const double pz = zPrev[ind];

                double displX = abs(xi - px);
                displX = displX > 0.5 * snapshotParams.interval.x
                             ? displX - snapshotParams.interval.x
                             : displX;
                double displY = abs(yi - py);
                displY = displY > 0.5 * snapshotParams.interval.y
                             ? displY - snapshotParams.interval.y
                             : displY;
                double displZ = abs(zi - pz);
                displZ = displZ > 0.5 * snapshotParams.interval.z
                             ? displZ - snapshotParams.interval.z
                             : displZ;

                file << xi;
                file << ",";
                file << yi;
                file << ",";
                file << zi;
                file << ",";
                file << snapshotParams.r[i];
                file << ",";
                file << vxi;
                file << ",";
                file << vyi;
                file << ",";
                file << vzi;
                file << ",";
                file << sqrt(vxi * vxi + vyi * vyi + vzi * vzi);
                file << ",";
                file << snapshotParams.vr[i];
                file << ",";
                file << snapshotParams.path[i];
                file << ",";
                file << snapshotParams.energy[i];
                file << ",";
                file << sqrt(displX * displX + displY * displY +
                             displZ * displZ);
                file << ",";
                file << snapshotParams.error[i];
                file << ",";
                file << ind;
                file << "\n";

                xPrev[ind] = xi;
                yPrev[ind] = yi;
                zPrev[ind] = zi;
            }
        }
    };

    // Spawn a new thread to write the snapshot to a file
    params.ioThread =
        std::thread(writeSnapshot, std::cref(params.snapshotParams),
                    params.hostData.numSnapshots++, params.previousX.data(),
                    params.previousY.data(), params.previousZ.data());
}

void end(Params &params) {
    printf("Cleaning up...\n");
    if (params.ioThread.joinable()) {
        params.ioThread.join();
    }

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(params.deviceConstants)));
    CUDA_CALL(cudaFree(params.memory));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinnedMemory)));

    CUDA_CALL(cudaEventDestroy(params.event1));
    CUDA_CALL(cudaEventDestroy(params.event2));

    CUDA_CALL(cudaStreamDestroy(params.stream2));
    CUDA_CALL(cudaStreamDestroy(params.stream1));
}

void init(const char *inputFileName, Params &params) {
    printf("==============\nInitialization\n==============\n");
    printf("Reading inputs from %s\n", inputFileName);
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

    params.hostData.addFlow = 1 == flow["impose"];
    params.hostConstants.flowLbb = flow["lbb"];
    params.hostConstants.flowTfr = flow["tfr"];
    params.hostConstants.flowVel = flow["velocity"];
    params.hostConstants.flowVel *= params.hostConstants.fZeroPerMuZero;

    params.hostData.errorTolerance = inputJson["errorTolerance"]["value"];
    params.hostData.snapshotFrequency = inputJson["snapShot"]["frequency"];
    params.snapshotParams.name = inputJson["snapShot"]["filename"];

    params.hostConstants.wallDragStrength = wall["drag"];
    params.hostConstants.xWall = 1 == wall["x"];
    params.hostConstants.yWall = 1 == wall["y"];
    params.hostConstants.zWall = 1 == wall["z"];
    params.hostConstants.dimensionality = box["dimensionality"];

    // Calculate the size of the box and the starting number of bubbles
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

    // Calculate the length of 'rows'.
    // Make it divisible by 32, as that's the warp size.
    params.bubbles.stride =
        params.bubbles.count +
        !!(params.bubbles.count % 32) * (32 - params.bubbles.count % 32);

    // It seems to hold that in 3 dimensions the total number of
    // bubble pairs is 11x and in two dimensions 4x number of bubbles.
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avgNumNeighbors =
        (params.hostConstants.dimensionality == 3) ? 11 : 4;
    params.pairs.stride = avgNumNeighbors * params.bubbles.stride;

    printf("---------------Starting parameters---------------\n");
    params.hostConstants.print();
    params.hostData.print();
    params.bubbles.print();
    params.pairs.print();
    printf("-------------------------------------------------\n");

    // Allocate and copy constants to GPU
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceConstants),
                           sizeof(Constants)));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceConstants),
                         static_cast<void *>(&params.hostConstants),
                         sizeof(Constants), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(dConstants,
                                 static_cast<void *>(&params.deviceConstants),
                                 sizeof(Constants *)));

    CUDA_ASSERT(cudaStreamCreate(&params.stream2));
    CUDA_ASSERT(cudaStreamCreate(&params.stream1));
    CUDA_CALL(cudaEventCreate(&params.event1, cudaEventDisableTiming));
    CUDA_CALL(cudaEventCreate(&params.event2, cudaEventDisableTiming));
    CUDA_CALL(
        cudaEventCreate(&params.snapshotParams.event, cudaEventDisableTiming));
    printRelevantInfoOfCurrentDevice();

    // Set device globals to zero
    int zero = 0;
    CUDA_CALL(
        cudaMemcpyToSymbol(dNumPairs, static_cast<void *>(&zero), sizeof(int)));
    KERNEL_LAUNCH(initGlobals, params, 0, 0);

    printf("Reserving device memory\n");
    CUDA_CALL(
        cudaMallocHost(&params.pinnedMemory, sizeof(int) + 3 * sizeof(double)));

    uint64_t bytes = params.bubbles.getMemReq();
    bytes += 2 * params.pairs.getMemReq();
    CUDA_ASSERT(cudaMalloc(&params.memory, bytes));

    // If we're going to be saving snapshots, allocate enough memory to hold all
    // the device data.
    if (0.0 < params.hostData.snapshotFrequency) {
        params.hostMemory.resize(bytes);
    }

    // Each named pointer is setup by these functions to point to
    // a different stride inside the continuous memory blob
    void *pairStart = params.bubbles.setupPointers(params.memory);
    pairStart = params.pairs.setupPointers(pairStart);
    params.setTempPointers(pairStart);

    params.previousX.resize(params.bubbles.stride);
    params.previousY.resize(params.bubbles.stride);
    params.previousZ.resize(params.bubbles.stride);
    params.snapshotParams.x0.resize(params.bubbles.stride);
    params.snapshotParams.y0.resize(params.bubbles.stride);
    params.snapshotParams.z0.resize(params.bubbles.stride);

    const uint64_t megs = bytes / (1024 * 1024);
    const uint64_t kilos = (bytes - megs * 1024 * 1024) / 1024;
    bytes = (bytes - megs * 1024 * 1024 - kilos * 1024);
    printf("Allocated %d MB %d KB %d B of global device memory.\n", megs, kilos,
           bytes);

    printf("Generating starting data\n");
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(
        generator, inputJson["rngSeed"]["value"]));
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
        generator, params.bubbles.r, params.bubbles.count,
        params.hostData.avgRad, bubbles["radius"]["std"]));
    CURAND_CALL(curandDestroyGenerator(generator));

    KERNEL_LAUNCH(assignDataToBubbles, params, 0, 0, bubblesPerDim,
                  params.hostData.avgRad, params.bubbles);

    // Get the average input surface area and maximum bubble radius
    void *cubPtr = static_cast<void *>(params.tempPair2);
    void *cubOutput = nullptr;
    void *out = static_cast<void *>(&params.hostConstants.averageSurfaceAreaIn);
    CUDA_CALL(cudaGetSymbolAddress(&cubOutput, dMaxRadius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cubPtr, params.pairs.getMemReq() / 2,
               params.bubbles.rp, static_cast<double *>(cubOutput),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(out, dMaxRadius, sizeof(double)));

    out = static_cast<void *>(&params.hostData.maxBubbleRadius);
    CUB_LAUNCH(&cub::DeviceReduce::Max, cubPtr, params.pairs.getMemReq() / 2,
               params.bubbles.r, static_cast<double *>(cubOutput),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(out, dMaxRadius, sizeof(double)));

    printf("First neighbor search\n");
    searchNeighbors(params);

    // After searchNeighbors x, y, z, r are correct,
    // but all predicted are trash. pairVelocity always uses
    // predicted values, so copy currents to predicteds
    bytes = params.bubbles.stride * sizeof(double);
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

    printf("Calculating initial velocities for Adams-Bashforth-Moulton\n");
    KERNEL_LAUNCH(
        resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
        params.bubbles.dxdto, params.bubbles.dydto, params.bubbles.dzdto,
        params.bubbles.drdto, params.bubbles.dxdtp, params.bubbles.dydtp,
        params.bubbles.dzdtp, params.bubbles.drdtp, params.bubbles.path);
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

    printf("Stabilizing a few rounds after creation\n");
    for (uint32_t i = 0; i < 5; ++i)
        stabilize(params, stabilizationSteps);

    printf("Scaling the simulation box\n");
    const double bubbleVolume = totalVolume(params);
    printf("Current phi: %.9g, target phi: %.9g\n",
           bubbleVolume / boxVolume(params), phi);

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
    params.snapshotParams.interval = params.hostConstants.interval;

    double mult = phi * boxVolume(params) / CUBBLE_PI;
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

    printf("Current phi: %.9g, target phi: %.9g\n",
           bubbleVolume / boxVolume(params), phi);

    printf("Neighbor search after scaling\n");
    searchNeighbors(params);
    // After searchNeighbors r is correct,
    // but rp is trash. pairVelocity always uses
    // predicted values, so copy r to rp
    bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDeviceToDevice, 0));

    printf("Stabilizing a few rounds after scaling\n");
    for (uint32_t i = 0; i < 5; ++i)
        stabilize(params, stabilizationSteps);

    printf("\n=============\nStabilization\n=============\n");
    params.hostData.numNeighborsSearched = 0;
    int numSteps = 0;
    const int failsafe = 500;

    printf("%-7s %-11s %-11s %-11s %-9s\n", "#steps", "dE", "e1", "e2",
           "#searches");
    while (true) {
        double time = stabilize(params, stabilizationSteps) *
                      params.hostData.timeScalingFactor;
        double deltaEnergy =
            std::abs(1.0 - params.hostData.energy1 / params.hostData.energy2) /
            time;

        if (deltaEnergy < inputJson["stabilization"]["maxDeltaEnergy"]) {
            printf("Final energies:");
            printf("\nbefore: %9.5e", params.hostData.energy1);
            printf("\nafter: %9.5e", params.hostData.energy2);
            printf("\ndelta: %9.5e", deltaEnergy);
            printf("\ntime: %9.5g\n", time);
            break;
        } else if (numSteps > failsafe) {
            printf("Over %d steps taken and required delta energy not reached. "
                   "Constraints might be too strict.\n");
            break;
        } else {
            printf("%-7d ", (numSteps + 1) * stabilizationSteps);
            printf("%-9.5e ", deltaEnergy);
            printf("%-9.5e ", params.hostData.energy1);
            printf("%-9.5e ", params.hostData.energy2);
            printf("%-9d\n", params.hostData.numNeighborsSearched);
            params.hostData.numNeighborsSearched = 0;
        }

        ++numSteps;
    }

    if (0.0 < params.hostData.snapshotFrequency) {
        // Set starting positions.
        // Avoiding batched copy, because the pointers might not be in order
        int *index = reinterpret_cast<int *>(params.hostMemory.data());
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousX.data()),
                             static_cast<void *>(params.bubbles.x),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousY.data()),
                             static_cast<void *>(params.bubbles.y),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousZ.data()),
                             static_cast<void *>(params.bubbles.z),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(index),
                             static_cast<void *>(params.bubbles.index),
                             sizeof(int) * params.bubbles.count,
                             cudaMemcpyDeviceToHost));

        for (uint64_t i = 0; i < params.bubbles.count; i++) {
            const int ind = index[i];
            params.snapshotParams.x0[ind] = params.previousX[i];
            params.snapshotParams.y0[ind] = params.previousY[i];
            params.snapshotParams.z0[ind] = params.previousZ[i];
        }

        // 'Previous' vectors are updated with the previous positions after
        // every snapshot, but since there is nothing previous to the first
        // snapshot, initialize them with the starting positions.
        std::copy(params.snapshotParams.x0.begin(),
                  params.snapshotParams.x0.end(), params.previousX.begin());
        std::copy(params.snapshotParams.y0.begin(),
                  params.snapshotParams.y0.end(), params.previousY.begin());
        std::copy(params.snapshotParams.z0.begin(),
                  params.snapshotParams.z0.end(), params.previousZ.begin());
    }

    // Reset wrap counts to 0
    // Avoiding batched memset, because the pointers might not be in order
    bytes = sizeof(int) * params.bubbles.stride;
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountX, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountY, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountZ, 0, bytes));

    // Reset errors since integration starts after this
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.bubbles.error);

    params.hostData.energy1 = totalEnergy(params);
    params.hostData.timeInteger = 0;
    params.hostData.timeFraction = 0.0;
    params.hostData.timesPrinted = 1;
    params.hostData.numIntegrationSteps = 0;
}
}; // namespace

namespace cubble {
void run(std::string &&inputFileName) {
    Params params;
    init(inputFileName.c_str(), params);

    if (params.hostData.snapshotFrequency > 0.0) {
        saveSnapshot(params);
    }

    printf("\n===========\nIntegration\n===========\n");
    printf("%-5s ", "T");
    printf("%-8s ", "phi");
    printf("%-6s ", "R");
    printf("%9s ", "#b   ");
    printf("%10s ", "#pairs");
    printf("%-6s ", "#steps");
    printf("%-9s ", "#searches");
    printf("%-11s ", "min ts");
    printf("%-11s ", "max ts");
    printf("%-11s \n", "avg ts");

    bool continueSimulation = true;
    double minTimestep = 9999999.9;
    double maxTimestep = -1.0;
    double avgTimestep = 0.0;
    bool resetErrors = false;
    double &ts = params.hostData.timeStep;
    const double minInterval =
        3 == params.hostConstants.dimensionality
            ? 0.5 * params.hostConstants.interval.getMinComponent()
            : 0.5 * (params.hostConstants.interval.x <
                             params.hostConstants.interval.y
                         ? params.hostConstants.interval.x
                         : params.hostConstants.interval.y);

    CUBBLE_PROFILE(true);
    while (continueSimulation) {
        CUBBLE_PROFILE(false);

        integrate(params);

        // Continue if there are more than the specified minimum number of
        // bubbles left in the simulation and if the largest bubble is smaller
        // than the simulation box in every dimension
        continueSimulation =
            params.bubbles.count > params.hostData.minNumBubbles &&
            params.hostData.maxBubbleRadius < minInterval;

        // Track timestep
        minTimestep = ts < minTimestep ? ts : minTimestep;
        maxTimestep = ts > maxTimestep ? ts : maxTimestep;
        avgTimestep += ts;

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
            // Define lambda for calculating averages of some values
            auto getAvg = [&params](double *p, Bubbles &bubbles) -> double {
                void *cubPtr = static_cast<void *>(params.tempPair2);
                void *cubOutput = nullptr;
                double total = 0.0;
                CUDA_CALL(cudaGetSymbolAddress(&cubOutput, dMaxRadius));
                CUB_LAUNCH(&cub::DeviceReduce::Sum, cubPtr,
                           params.pairs.getMemReq() / 2, p,
                           static_cast<double *>(cubOutput),
                           params.bubbles.count, (cudaStream_t)0, false);
                CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total),
                                               dMaxRadius, sizeof(double)));

                return total / bubbles.count;
            };

            params.hostData.energy2 = totalEnergy(params);
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
                printf("Couldn't open file stream to append results to!\n");
            }

            const double phi = totalVolume(params) / boxVolume(params);

            printf("%-5d ", params.hostData.timesPrinted);
            printf("%-#8.6g ", phi);
            printf("%-#6.4g ", relRad);
            printf("%9d ", params.bubbles.count);
            printf("%10d ", params.pairs.count);
            printf("%6d ", params.hostData.numStepsInTimeStep);
            printf("%-9d ", params.hostData.numNeighborsSearched);
            printf("%-9.5e ", minTimestep);
            printf("%-9.5e ", maxTimestep);
            printf("%-9.5e \n",
                   avgTimestep / params.hostData.numStepsInTimeStep);

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

            const bool isSnapshotTime =
                params.hostData.timeInteger > nextSnapshotTimeInteger ||
                (params.hostData.timeInteger == nextSnapshotTimeInteger &&
                 params.hostData.timeFraction >= nextSnapshotTimeFraction);

            if (isSnapshotTime) {
                saveSnapshot(params);
            }
        }

        if (resetErrors) {
            KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count,
                          false, params.bubbles.error);
            resetErrors = false;
        }

        ++params.hostData.numStepsInTimeStep;
    }

    if (params.bubbles.count <= params.hostData.minNumBubbles) {
        printf("Stopping simulation, since the number of bubbles left in the "
               "simulation (%d) is less than or equal to the specified minimum "
               "(%d)\n",
               params.bubbles.count, params.hostData.minNumBubbles);
    } else if (params.hostData.maxBubbleRadius > minInterval) {
        dvec temp = params.hostConstants.interval;
        printf("Stopping simulation, since the radius of the largest bubble "
               "(%g) is greater than the simulation box (%g, %g, %g)\n",
               params.hostData.maxBubbleRadius, temp.x, temp.y, temp.z);
    } else {
        printf("Stopping simulation for an unknown reason...\n");
    }

    if (params.hostData.snapshotFrequency > 0.0) {
        saveSnapshot(params);
    }

    end(params);
    printf("Done\n");
}
} // namespace cubble
