#include "CubWrapper.h"
#include "DataDefinitions.h"
#include "Kernels.cuh"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include "nlohmann/json.hpp"
#include <array>
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

void updateCellsAndNeighbors(Params &params) {
    NVTX_RANGE_PUSH_A("Neighbors");
    params.hostData.numNeighborsSearched++;
    // Boundary wrap
    KERNEL_LAUNCH(wrapKernel, params.pairKernelSize, 0, 0,
                  params.hostData.numBubbles, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_Z]);

    // Update saved values
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_X]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::X]),
        sizeof(double) * params.hostData.dataStride, cudaMemcpyDeviceToDevice,
        0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_Y]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::Y]),
        sizeof(double) * params.hostData.dataStride, cudaMemcpyDeviceToDevice,
        0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_Z]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::Z]),
        sizeof(double) * params.hostData.dataStride, cudaMemcpyDeviceToDevice,
        0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_R]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::R]),
        sizeof(double) * params.hostData.dataStride, cudaMemcpyDeviceToDevice,
        0));

    // Minimum size of cell is twice the sum of the skin and max bubble radius
    ivec cellDim = (params.hostConstants.interval /
                    (2 * (params.hostData.maxBubbleRadius +
                          params.hostConstants.skinRadius)))
                       .floor();
    cellDim.z = cellDim.z > 0 ? cellDim.z : 1;
    dim3 gridSize = dim3(cellDim.x, cellDim.y, cellDim.z);

    // Determine the maximum number of Morton numbers for the simulation box
    const int maxGridSize =
        gridSize.x > gridSize.y
            ? (gridSize.x > gridSize.z ? gridSize.x : gridSize.z)
            : (gridSize.y > gridSize.z ? gridSize.y : gridSize.z);
    int maxNumCells = 1;
    while (maxNumCells < maxGridSize)
        maxNumCells = maxNumCells << 1;

    if (NUM_DIM == 3)
        maxNumCells = maxNumCells * maxNumCells * maxNumCells;
    else
        maxNumCells = maxNumCells * maxNumCells;

#ifndef NDEBUG
    std::cout << "Max num cells: " << maxNumCells << ", grid size: ("
              << gridSize.x << ", " << gridSize.y << ", " << gridSize.z
              << "), avg num bubbles per cell: "
              << params.hostData.numBubbles /
                     (gridSize.x * gridSize.y * gridSize.z)
              << std::endl;
#endif
    assert(maxNumCells < params.hostData.pairStride);

    int *offsets = params.dips[(uint32_t)DIP::PAIR1];
    int *sizes = params.dips[(uint32_t)DIP::PAIR1] + maxNumCells;
    int *cellIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 0 * params.hostData.dataStride;
    int *bubbleIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 1 * params.hostData.dataStride;
    int *sortedCellIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 2 * params.hostData.dataStride;
    int *sortedBubbleIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 3 * params.hostData.dataStride;

    const uint64_t resetBytes =
        sizeof(int) * params.hostData.pairStride *
        ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

    // Reset number of neighbors to zero as they will be calculated again
    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::NUM_NEIGHBORS], 0,
                         sizeof(int) * params.hostData.dataStride));

    KERNEL_LAUNCH(assignBubblesToCells, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                  cellDim, params.hostData.numBubbles);

    params.cw.sortPairs<int, int>(
        &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices),
        sortedCellIndices, const_cast<const int *>(bubbleIndices),
        sortedBubbleIndices, params.hostData.numBubbles);

    params.cw.histogram<int *, int, int, int>(
        &cub::DeviceHistogram::HistogramEven, cellIndices, sizes,
        maxNumCells + 1, 0, maxNumCells, params.hostData.numBubbles);

    params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets,
                                 maxNumCells);

    auto copyAndSwap = [](Params &params, int *inds, auto &&arr, uint32_t from,
                          uint32_t to) {
        KERNEL_LAUNCH(copyKernel, params.defaultKernelSize, 0, 0,
                      params.hostData.numBubbles,
                      ReorganizeType::COPY_FROM_INDEX, inds, inds, arr[from],
                      arr[to]);

        auto *swapper = arr[from];
        arr[from] = arr[to];
        arr[to] = swapper;
    };

    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::X,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::Y,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::Z,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::R,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DXDT,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DYDT,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DZDT,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DRDT,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DXDTO,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DYDTO,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DZDTO,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::DRDTO,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::X0,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::Y0,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::Z0,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::PATH,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps,
                (uint32_t)DDP::DISTANCE, (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps,
                (uint32_t)DDP::SAVED_X, (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps,
                (uint32_t)DDP::SAVED_Y, (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps,
                (uint32_t)DDP::SAVED_Z, (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps,
                (uint32_t)DDP::SAVED_R, (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.ddps, (uint32_t)DDP::ERROR,
                (uint32_t)DDP::TEMP_DATA);
    copyAndSwap(params, sortedBubbleIndices, params.dips,
                (uint32_t)DIP::WRAP_COUNT_X, (uint32_t)DIP::TEMP);
    copyAndSwap(params, sortedBubbleIndices, params.dips,
                (uint32_t)DIP::WRAP_COUNT_Y, (uint32_t)DIP::TEMP);
    copyAndSwap(params, sortedBubbleIndices, params.dips,
                (uint32_t)DIP::WRAP_COUNT_Z, (uint32_t)DIP::TEMP);
    copyAndSwap(params, sortedBubbleIndices, params.dips, (uint32_t)DIP::INDEX,
                (uint32_t)DIP::TEMP);

    KernelSize kernelSizeNeighbor = KernelSize(gridSize, dim3(128, 1, 1));

    int *dnp = nullptr;
    CUDA_ASSERT(
        cudaGetSymbolAddress(reinterpret_cast<void **>(&dnp), dNumPairs));
    CUDA_CALL(cudaMemset(dnp, 0, sizeof(int)));

    for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i) {
        cudaStream_t stream =
            (i % 2) ? params.velocityStream : params.gasStream;
        KERNEL_LAUNCH(
            neighborSearch, kernelSizeNeighbor, 0, stream, i,
            params.hostData.numBubbles, maxNumCells,
            (int)params.hostData.pairStride, offsets, sizes,
            params.dips[(uint32_t)DIP::PAIR1COPY],
            params.dips[(uint32_t)DIP::PAIR2COPY],
            params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::X],
            params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
            params.dips[(uint32_t)DIP::NUM_NEIGHBORS]);
    }

    CUDA_CALL(cudaMemcpy(static_cast<void *>(&params.hostData.numPairs),
                         static_cast<void *>(dnp), sizeof(int),
                         cudaMemcpyDeviceToHost));

#ifndef NDEBUG
    std::cout << "Max num pairs: " << params.hostData.pairStride
              << ", actual num pairs: " << params.hostData.numPairs
              << std::endl;
#endif

    params.cw.sortPairs<int, int>(
        &cub::DeviceRadixSort::SortPairs,
        const_cast<const int *>(params.dips[(uint32_t)DIP::PAIR1COPY]),
        params.dips[(uint32_t)DIP::PAIR1],
        const_cast<const int *>(params.dips[(uint32_t)DIP::PAIR2COPY]),
        params.dips[(uint32_t)DIP::PAIR2], params.hostData.numPairs);
    NVTX_RANGE_POP();
}

void deleteSmallBubbles(Params &params, int numToBeDeleted) {
    NVTX_RANGE_PUSH_A("BubbleRemoval");

    KERNEL_LAUNCH(
        swapDataCountPairs, params.pairKernelSize, 0, 0,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.dips[(uint32_t)DIP::TEMP],
        params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DYDT],
        params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DRDT],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
        params.ddps[(uint32_t)DDP::DZDTO], params.ddps[(uint32_t)DDP::DRDTO],
        params.ddps[(uint32_t)DDP::X0], params.ddps[(uint32_t)DDP::Y0],
        params.ddps[(uint32_t)DDP::Z0], params.ddps[(uint32_t)DDP::PATH],
        params.ddps[(uint32_t)DDP::DISTANCE],
        params.ddps[(uint32_t)DDP::SAVED_X],
        params.ddps[(uint32_t)DDP::SAVED_Y],
        params.ddps[(uint32_t)DDP::SAVED_Z],
        params.ddps[(uint32_t)DDP::SAVED_R], params.ddps[(uint32_t)DDP::ERROR],
        params.dips[(uint32_t)DIP::WRAP_COUNT_X],
        params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
        params.dips[(uint32_t)DIP::WRAP_COUNT_Z],
        params.dips[(uint32_t)DIP::INDEX],
        params.dips[(uint32_t)DIP::NUM_NEIGHBORS]);

    KERNEL_LAUNCH(addVolumeFixPairs, params.pairKernelSize, 0, 0,
                  params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                  params.dips[(uint32_t)DIP::PAIR2],
                  params.dips[(uint32_t)DIP::TEMP],
                  params.ddps[(uint32_t)DDP::R]);

    // Update kernel sizes based on number of remaining bubbles
    params.hostData.numBubbles -= numToBeDeleted;
    params.defaultKernelSize = KernelSize(128, params.hostData.numBubbles);
    int numBlocks =
        std::min(1024, (int)std::ceil(params.hostData.numBubbles / 128.0));
    params.pairKernelSize = KernelSize(dim3(numBlocks, 1, 1), dim3(128, 1, 1));

    NVTX_RANGE_POP();
}

void saveSnapshotToFile(Params &params) {
    // Calculate total energy
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.hostData.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(
        potentialEnergyKernel, params.pairKernelSize, 0, 0,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z]);

    std::stringstream ss;
    ss << "snapshot.csv." << params.hostData.numSnapshots;
    std::ofstream file(ss.str().c_str(), std::ios::out);
    if (file.is_open()) {
        std::vector<double> doubleData;
        doubleData.resize(params.hostData.dataStride *
                          (uint32_t)DDP::NUM_VALUES);
        for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i) {
            CUDA_CALL(cudaMemcpy(&doubleData[i * params.hostData.dataStride],
                                 params.ddps[i],
                                 sizeof(double) * params.hostData.dataStride,
                                 cudaMemcpyDeviceToHost));
        }

        std::vector<int> intData;
        intData.resize(params.hostData.dataStride);
        CUDA_CALL(cudaMemcpy(intData.data(), params.dips[(uint32_t)DIP::INDEX],
                             sizeof(intData[0]) * intData.size(),
                             cudaMemcpyDeviceToHost));

        if (params.hostData.numSnapshots == 0) {
            for (uint64_t i = 0; i < (uint64_t)params.hostData.numBubbles;
                 ++i) {
                params.previousX[intData[i]] =
                    doubleData[i + 0 * params.hostData.dataStride];
                params.previousY[intData[i]] =
                    doubleData[i + 1 * params.hostData.dataStride];
                params.previousZ[intData[i]] =
                    doubleData[i + 2 * params.hostData.dataStride];
            }
        }

        file << "x,y,z,r,vx,vy,vz,vtot,vr,path,distance,energy,displacement,"
                "error,index\n ";
        for (uint64_t i = 0; i < (uint64_t)params.hostData.numBubbles; ++i) {
            const double x =
                doubleData[i + (uint32_t)DDP::X * params.hostData.dataStride];
            const double y =
                doubleData[i + (uint32_t)DDP::Y * params.hostData.dataStride];
            const double z =
                doubleData[i + (uint32_t)DDP::Z * params.hostData.dataStride];
            const double r =
                doubleData[i + (uint32_t)DDP::R * params.hostData.dataStride];
            const double vx = doubleData[i + (uint32_t)DDP::DXDT *
                                                 params.hostData.dataStride];
            const double vy = doubleData[i + (uint32_t)DDP::DYDT *
                                                 params.hostData.dataStride];
            const double vz = doubleData[i + (uint32_t)DDP::DZDT *
                                                 params.hostData.dataStride];
            const double vr = doubleData[i + (uint32_t)DDP::DRDT *
                                                 params.hostData.dataStride];
            const double path = doubleData[i + (uint32_t)DDP::PATH *
                                                   params.hostData.dataStride];
            const double distance =
                doubleData[i + (uint32_t)DDP::DISTANCE *
                                   params.hostData.dataStride];
            const double error = doubleData[i + (uint32_t)DDP::ERROR *
                                                    params.hostData.dataStride];
            const double energy =
                doubleData[i + (uint32_t)DDP::TEMP_DATA *
                                   params.hostData.dataStride];
            const double px = params.previousX[intData[i]];
            const double py = params.previousY[intData[i]];
            const double pz = params.previousZ[intData[i]];

            double displX = abs(x - px);
            displX = displX > 0.5 * params.hostConstants.interval.x
                         ? displX - params.hostConstants.interval.x
                         : displX;
            double displY = abs(y - py);
            displY = displY > 0.5 * params.hostConstants.interval.y
                         ? displY - params.hostConstants.interval.y
                         : displY;
            double displZ = abs(z - pz);
            displZ = displZ > 0.5 * params.hostConstants.interval.z
                         ? displZ - params.hostConstants.interval.z
                         : displZ;

            file << x;
            file << ",";
            file << y;
            file << ",";
            file << z;
            file << ",";
            file << r;
            file << ",";
            file << vx;
            file << ",";
            file << vy;
            file << ",";
            file << vz;
            file << ",";
            file << sqrt(vx * vx + vy * vy + vz * vz);
            file << ",";
            file << vr;
            file << ",";
            file << path;
            file << ",";
            file << distance;
            file << ",";
            file << energy;
            file << ",";
            file << sqrt(displX * displX + displY * displY + displZ * displZ);
            file << ",";
            file << error;
            file << ",";
            file << intData[i + 0 * params.hostData.dataStride];
            file << "\n";

            params.previousX[intData[i]] = x;
            params.previousY[intData[i]] = y;
            params.previousZ[intData[i]] = z;
        }

        ++params.hostData.numSnapshots;
    }
}

double stabilize(Params &params, int numStepsToRelax) {
    // This function integrates only the positions of the bubbles.
    // Gas exchange is not used. This is used for equilibrating the foam.

    double elapsedTime = 0.0;
    double error = 100000;

    // Energy before stabilization
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.hostData.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(
        potentialEnergyKernel, params.pairKernelSize, 0, 0,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z]);

    params.hostData.energy1 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.hostData.numBubbles);

    for (int i = 0; i < numStepsToRelax; ++i) {
        do {
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.hostData.numBubbles,
                          params.ddps[(uint32_t)DDP::DXDTP],
                          params.ddps[(uint32_t)DDP::DYDTP],
                          params.ddps[(uint32_t)DDP::DZDTP]);

            KERNEL_LAUNCH(
                predictKernel, params.pairKernelSize, 0, 0,
                params.hostData.numBubbles, params.hostData.timeStep, false,
                params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                params.ddps[(uint32_t)DDP::DXDT],
                params.ddps[(uint32_t)DDP::DXDTO],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::DYDT],
                params.ddps[(uint32_t)DDP::DYDTO],
                params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
                params.ddps[(uint32_t)DDP::DZDT],
                params.ddps[(uint32_t)DDP::DZDTO],
                params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::DRDT],
                params.ddps[(uint32_t)DDP::DRDTO]);

            KERNEL_LAUNCH(
                velocityPairKernel, params.pairKernelSize, 0, 0,
                params.dips[(uint32_t)DIP::PAIR1],
                params.dips[(uint32_t)DIP::PAIR2],
                params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::XP],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::DXDTP],
                params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDTP]);

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
            KERNEL_LAUNCH(
                velocityWallKernel, params.pairKernelSize, 0,
                params.velocityStream, params.hostData.numBubbles,
                params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::XP],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::DXDTP],
                params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDTP]);
#endif
            // Correct
            KERNEL_LAUNCH(
                correctKernel, params.pairKernelSize, 0, 0,
                params.hostData.numBubbles, params.hostData.timeStep, false,
                params.ddps[(uint32_t)DDP::ERROR],
                params.ddps[(uint32_t)DDP::TEMP_DATA],
                params.dips[(uint32_t)DIP::TEMP],
                params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                params.ddps[(uint32_t)DDP::DXDT],
                params.ddps[(uint32_t)DDP::DXDTP],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::DYDT],
                params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
                params.ddps[(uint32_t)DDP::DZDT],
                params.ddps[(uint32_t)DDP::DZDTP],
                params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::DRDT],
                params.ddps[(uint32_t)DDP::DRDTP],
                params.ddps[(uint32_t)DDP::SAVED_X],
                params.ddps[(uint32_t)DDP::SAVED_Y],
                params.ddps[(uint32_t)DDP::SAVED_Z],
                params.ddps[(uint32_t)DDP::SAVED_R]);

            KERNEL_LAUNCH(endStepKernel, params.pairKernelSize, 0,
                          params.gasStream, params.hostData.numBubbles,
                          params.ddps[(uint32_t)DDP::TEMP_DATA],
                          params.ddps[(uint32_t)DDP::SAVED_X],
                          params.ddps[(uint32_t)DDP::SAVED_Y],
                          params.ddps[(uint32_t)DDP::SAVED_Z],
                          params.ddps[(uint32_t)DDP::SAVED_R],
                          (int)params.pairKernelSize.grid.x);

            CUDA_CALL(cudaMemcpyAsync(
                static_cast<void *>(params.pinnedDouble),
                params.ddps[(uint32_t)DDP::TEMP_DATA], 3 * sizeof(double),
                cudaMemcpyDeviceToHost, params.gasStream));

            CUDA_CALL(cudaEventRecord(params.event1, params.gasStream));

            // Error
            // Wait for event
            CUDA_CALL(cudaEventSynchronize(params.event1));
            error = params.pinnedDouble[0];

            if (error < params.hostData.errorTolerance &&
                params.hostData.timeStep < 0.1)
                params.hostData.timeStep *= 1.9;
            else if (error > params.hostData.errorTolerance)
                params.hostData.timeStep *= 0.5;

        } while (error > params.hostData.errorTolerance);

        // Update the current values with the calculated predictions
        double *swapper = params.ddps[(uint32_t)DDP::DXDTO];
        params.ddps[(uint32_t)DDP::DXDTO] = params.ddps[(uint32_t)DDP::DXDT];
        params.ddps[(uint32_t)DDP::DXDT] = params.ddps[(uint32_t)DDP::DXDTP];
        params.ddps[(uint32_t)DDP::DXDTP] = swapper;

        swapper = params.ddps[(uint32_t)DDP::DYDTO];
        params.ddps[(uint32_t)DDP::DYDTO] = params.ddps[(uint32_t)DDP::DYDT];
        params.ddps[(uint32_t)DDP::DYDT] = params.ddps[(uint32_t)DDP::DYDTP];
        params.ddps[(uint32_t)DDP::DYDTP] = swapper;

        swapper = params.ddps[(uint32_t)DDP::DZDTO];
        params.ddps[(uint32_t)DDP::DZDTO] = params.ddps[(uint32_t)DDP::DZDT];
        params.ddps[(uint32_t)DDP::DZDT] = params.ddps[(uint32_t)DDP::DZDTP];
        params.ddps[(uint32_t)DDP::DZDTP] = swapper;

        swapper = params.ddps[(uint32_t)DDP::X];
        params.ddps[(uint32_t)DDP::X] = params.ddps[(uint32_t)DDP::XP];
        params.ddps[(uint32_t)DDP::XP] = swapper;

        swapper = params.ddps[(uint32_t)DDP::Y];
        params.ddps[(uint32_t)DDP::Y] = params.ddps[(uint32_t)DDP::YP];
        params.ddps[(uint32_t)DDP::YP] = swapper;

        swapper = params.ddps[(uint32_t)DDP::Z];
        params.ddps[(uint32_t)DDP::Z] = params.ddps[(uint32_t)DDP::ZP];
        params.ddps[(uint32_t)DDP::ZP] = swapper;

        elapsedTime += params.hostData.timeStep;

        if (2 * params.pinnedDouble[2] >= params.hostConstants.skinRadius) {
            updateCellsAndNeighbors(params);
        }
    }

    // Energy after stabilization
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.hostData.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(
        potentialEnergyKernel, params.pairKernelSize, 0, 0,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z]);

    params.hostData.energy2 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.hostData.numBubbles);

    return elapsedTime;
}

void velocityCalculation(Params &params) {
    // Velocity
    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, params.velocityStream,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
        params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::XP],
        params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
        params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
        params.ddps[(uint32_t)DDP::DZDTP]);

    // Flow velocity
#if (USE_FLOW == 1)
    {
        KERNEL_LAUNCH(neighborVelocityKernel, params.pairKernelSize, 0,
                      params.velocityStream, params.dips[(uint32_t)DIP::PAIR1],
                      params.dips[(uint32_t)DIP::PAIR2],
                      params.ddps[(uint32_t)DDP::FLOW_VX],
                      params.ddps[(uint32_t)DDP::FLOW_VY],
                      params.ddps[(uint32_t)DDP::FLOW_VZ],
                      params.ddps[(uint32_t)DDP::DXDTO],
                      params.ddps[(uint32_t)DDP::DYDTO],
                      params.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(
            flowVelocityKernel, params.pairKernelSize, 0, params.velocityStream,
            params.hostData.numBubbles,
            params.dips[(uint32_t)DIP::NUM_NEIGHBORS],
            params.ddps[(uint32_t)DDP::DXDTP],
            params.ddps[(uint32_t)DDP::DYDTP],
            params.ddps[(uint32_t)DDP::DZDTP],
            params.ddps[(uint32_t)DDP::FLOW_VX],
            params.ddps[(uint32_t)DDP::FLOW_VY],
            params.ddps[(uint32_t)DDP::FLOW_VZ], params.ddps[(uint32_t)DDP::XP],
            params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
            params.ddps[(uint32_t)DDP::RP]);
    }
#endif

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    // Wall velocity, should be after flow so that possible drag is applied
    // correctly
    KERNEL_LAUNCH(
        velocityWallKernel, params.pairKernelSize, 0, params.velocityStream,
        params.hostData.numBubbles, params.ddps[(uint32_t)DDP::RP],
        params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
        params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
        params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP]);
#endif
}

void gasExchangeCalculation(Params &params) {
    // Gas exchange
    KERNEL_LAUNCH(
        gasExchangeKernel, params.pairKernelSize, 0, params.gasStream,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::RP],
        params.ddps[(uint32_t)DDP::DRDTP],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::XP],
        params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP]);

    KERNEL_LAUNCH(
        finalRadiusChangeRateKernel, params.pairKernelSize, 0, params.gasStream,
        params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::RP],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.hostData.numBubbles);
}

bool integrate(Params &params) {
    NVTX_RANGE_PUSH_A("Integration function");

    double error = 100000;
    uint32_t numLoopsDone = 0;

    do {
        NVTX_RANGE_PUSH_A("Integration step");

        // Reset
        KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0,
                      params.velocityStream, 0.0, params.hostData.numBubbles,
                      params.ddps[(uint32_t)DDP::DXDTP],
                      params.ddps[(uint32_t)DDP::DYDTP],
                      params.ddps[(uint32_t)DDP::DZDTP],
                      params.ddps[(uint32_t)DDP::DRDTP],
                      params.ddps[(uint32_t)DDP::TEMP_DATA],
                      params.ddps[(uint32_t)DDP::FLOW_VX],
                      params.ddps[(uint32_t)DDP::FLOW_VY],
                      params.ddps[(uint32_t)DDP::FLOW_VZ]);

        // Predict
        KERNEL_LAUNCH(
            predictKernel, params.pairKernelSize, 0, params.gasStream,
            params.hostData.numBubbles, params.hostData.timeStep, true,
            params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
            params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTO],
            params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
            params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTO],
            params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
            params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTO],
            params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::R],
            params.ddps[(uint32_t)DDP::DRDT],
            params.ddps[(uint32_t)DDP::DRDTO]);

        CUDA_CALL(cudaEventRecord(params.event1, params.gasStream));
        gasExchangeCalculation(params);
        CUDA_CALL(cudaStreamWaitEvent(params.velocityStream, params.event1, 0));
        velocityCalculation(params);

        // Correct
        KERNEL_LAUNCH(
            correctKernel, params.pairKernelSize, 0, 0,
            params.hostData.numBubbles, params.hostData.timeStep, true,
            params.ddps[(uint32_t)DDP::ERROR],
            params.ddps[(uint32_t)DDP::FLOW_VX],
            params.dips[(uint32_t)DIP::TEMP], params.ddps[(uint32_t)DDP::XP],
            params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT],
            params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::YP],
            params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDT],
            params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::ZP],
            params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDT],
            params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::RP],
            params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::DRDT],
            params.ddps[(uint32_t)DDP::DRDTP],
            params.ddps[(uint32_t)DDP::SAVED_X],
            params.ddps[(uint32_t)DDP::SAVED_Y],
            params.ddps[(uint32_t)DDP::SAVED_Z],
            params.ddps[(uint32_t)DDP::SAVED_R]);

        // Copy numToBeDeleted to pinned memory
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.pinnedInt),
                                  static_cast<void *>(params.numToBeDeleted),
                                  sizeof(int), cudaMemcpyDeviceToHost,
                                  params.gasStream));

        KERNEL_LAUNCH(endStepKernel, params.pairKernelSize, 0,
                      params.velocityStream, params.hostData.numBubbles,
                      params.ddps[(uint32_t)DDP::FLOW_VX],
                      params.ddps[(uint32_t)DDP::SAVED_X],
                      params.ddps[(uint32_t)DDP::SAVED_Y],
                      params.ddps[(uint32_t)DDP::SAVED_Z],
                      params.ddps[(uint32_t)DDP::SAVED_R],
                      (int)params.pairKernelSize.grid.x);

        // Copy maximum error, maximum radius and maximum boundary expansion to
        // pinned memory. See correctKernel and endStepKernel for details.
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.pinnedDouble),
                                  params.ddps[(uint32_t)DDP::FLOW_VX],
                                  3 * sizeof(double), cudaMemcpyDeviceToHost,
                                  params.velocityStream));

        CUDA_CALL(cudaEventRecord(params.event1, params.velocityStream));

        // Path lenghts & distances
        KERNEL_LAUNCH(
            pathLengthDistanceKernel, params.pairKernelSize, 0,
            params.gasStream, params.hostData.numBubbles,
            params.ddps[(uint32_t)DDP::TEMP_DATA],
            params.ddps[(uint32_t)DDP::PATH],
            params.ddps[(uint32_t)DDP::DISTANCE],
            params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
            params.ddps[(uint32_t)DDP::X0],
            params.dips[(uint32_t)DIP::WRAP_COUNT_X],
            params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
            params.ddps[(uint32_t)DDP::Y0],
            params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
            params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
            params.ddps[(uint32_t)DDP::Z0],
            params.dips[(uint32_t)DIP::WRAP_COUNT_Z]);

        // Wait for event
        CUDA_CALL(cudaEventSynchronize(params.event1));

        error = params.pinnedDouble[0];
        if (error < params.hostData.errorTolerance &&
            params.hostData.timeStep < 0.1)
            params.hostData.timeStep *= 1.9;
        else if (error > params.hostData.errorTolerance)
            params.hostData.timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > params.hostData.errorTolerance);

    // Update values
    double *swapper = params.ddps[(uint32_t)DDP::DXDTO];
    params.ddps[(uint32_t)DDP::DXDTO] = params.ddps[(uint32_t)DDP::DXDT];
    params.ddps[(uint32_t)DDP::DXDT] = params.ddps[(uint32_t)DDP::DXDTP];
    params.ddps[(uint32_t)DDP::DXDTP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::DYDTO];
    params.ddps[(uint32_t)DDP::DYDTO] = params.ddps[(uint32_t)DDP::DYDT];
    params.ddps[(uint32_t)DDP::DYDT] = params.ddps[(uint32_t)DDP::DYDTP];
    params.ddps[(uint32_t)DDP::DYDTP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::DZDTO];
    params.ddps[(uint32_t)DDP::DZDTO] = params.ddps[(uint32_t)DDP::DZDT];
    params.ddps[(uint32_t)DDP::DZDT] = params.ddps[(uint32_t)DDP::DZDTP];
    params.ddps[(uint32_t)DDP::DZDTP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::DRDTO];
    params.ddps[(uint32_t)DDP::DRDTO] = params.ddps[(uint32_t)DDP::DRDT];
    params.ddps[(uint32_t)DDP::DRDT] = params.ddps[(uint32_t)DDP::DRDTP];
    params.ddps[(uint32_t)DDP::DRDTP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::X];
    params.ddps[(uint32_t)DDP::X] = params.ddps[(uint32_t)DDP::XP];
    params.ddps[(uint32_t)DDP::XP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::Y];
    params.ddps[(uint32_t)DDP::Y] = params.ddps[(uint32_t)DDP::YP];
    params.ddps[(uint32_t)DDP::YP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::Z];
    params.ddps[(uint32_t)DDP::Z] = params.ddps[(uint32_t)DDP::ZP];
    params.ddps[(uint32_t)DDP::ZP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::R];
    params.ddps[(uint32_t)DDP::R] = params.ddps[(uint32_t)DDP::RP];
    params.ddps[(uint32_t)DDP::RP] = swapper;

    swapper = params.ddps[(uint32_t)DDP::PATH];
    params.ddps[(uint32_t)DDP::PATH] = params.ddps[(uint32_t)DDP::TEMP_DATA];
    params.ddps[(uint32_t)DDP::TEMP_DATA] = swapper;

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

    params.hostData.maxBubbleRadius = params.pinnedDouble[1];

    // Delete, if there are nonzero amount of bubbles with a radius
    // smaller than the minimum radius. See correctKernel for the
    // comparison & calculation.
    if (params.pinnedInt[0] > 0) {
        deleteSmallBubbles(params, params.pinnedInt[0]);
    }

    // If the boundary of the bubble with maximum sum of movement & expansion
    // has moved more than half of the "skin radius", reorder bubbles.
    // See correctKernel, comparePair for details.
    if (params.pinnedDouble[2] >= 0.5 * params.hostConstants.skinRadius) {
        updateCellsAndNeighbors(params);
    }

    bool continueSimulation =
        params.hostData.numBubbles > params.hostData.minNumBubbles;
    continueSimulation &=
        (NUM_DIM == 3)
            ? params.hostData.maxBubbleRadius <
                  0.5 * params.hostConstants.interval.getMinComponent()
            : true;

    NVTX_RANGE_POP();

    return continueSimulation;
}

void transformPositions(Params &params, bool normalize) {
    KERNEL_LAUNCH(transformPositionsKernel, params.pairKernelSize, 0, 0,
                  normalize, params.hostData.numBubbles,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z]);
}

double calculateVolumeOfBubbles(Params &params) {
    KERNEL_LAUNCH(calculateVolumes, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA],
                  params.hostData.numBubbles);

    return params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.hostData.numBubbles);
}

void deinit(Params &params) {
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(params.deviceDoubleMemory)));
    CUDA_CALL(cudaFree(static_cast<void *>(params.deviceIntMemory)));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinnedInt)));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinnedDouble)));

    CUDA_CALL(cudaEventDestroy(params.event1));

    CUDA_CALL(cudaStreamDestroy(params.velocityStream));
    CUDA_CALL(cudaStreamDestroy(params.gasStream));
}

double getSimulationBoxVolume(Params &params) {
    dvec temp = params.hostConstants.interval;
    return (NUM_DIM == 3) ? temp.x * temp.y * temp.z : temp.x * temp.y;
}

void commonSetup(Params &params) {
    params.defaultKernelSize = KernelSize(128, params.hostData.numBubbles);

    // Streams
    CUDA_ASSERT(cudaStreamCreate(&params.velocityStream));
    CUDA_ASSERT(cudaStreamCreate(&params.gasStream));

    printRelevantInfoOfCurrentDevice();

    CUDA_CALL(cudaEventCreate(&params.event1));

    CUDA_CALL(cudaGetSymbolAddress(
        reinterpret_cast<void **>(&params.numToBeDeleted), dNumToBeDeleted));

    // Set device globals to zero
    double zero = 0.0;
    int zeroI = 0;
    bool falseB = false;
    CUDA_CALL(cudaMemcpyToSymbol(dTotalArea, reinterpret_cast<void *>(&zero),
                                 sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dTotalOverlapArea, reinterpret_cast<void *>(&zero), sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dTotalOverlapAreaPerRadius,
                                 reinterpret_cast<void *>(&zero),
                                 sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dTotalAreaPerRadius, reinterpret_cast<void *>(&zero), sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dTotalVolumeNew, reinterpret_cast<void *>(&zero), sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dTotalVolumeOld, reinterpret_cast<void *>(&zero), sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dErrorEncountered, reinterpret_cast<void *>(&falseB), sizeof(bool)));
    CUDA_CALL(cudaMemcpyToSymbol(dNumPairs, reinterpret_cast<void *>(&zeroI),
                                 sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(dNumPairsNew, reinterpret_cast<void *>(&zeroI),
                                 sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dNumToBeDeleted, reinterpret_cast<void *>(&zeroI), sizeof(int)));

    std::cout << "Reserving device memory to hold data." << std::endl;

    CUDA_CALL(cudaMallocHost(reinterpret_cast<void **>(&params.pinnedDouble),
                             sizeof(double) * 3));
    CUDA_CALL(cudaMallocHost(reinterpret_cast<void **>(&params.pinnedInt),
                             sizeof(int)));

    // Calculate the length of 'rows'. Will be divisible by 32, as that's the
    // warp size.
    params.hostData.dataStride =
        params.hostData.numBubbles + !!(params.hostData.numBubbles % 32) *
                                         (32 - params.hostData.numBubbles % 32);

    // Doubles
    params.hostData.memReqD = sizeof(double) *
                              (uint64_t)params.hostData.dataStride *
                              (uint64_t)DDP::NUM_VALUES;
    CUDA_ASSERT(
        cudaMalloc(reinterpret_cast<void **>(&params.deviceDoubleMemory),
                   params.hostData.memReqD));

    for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
        params.ddps[i] =
            params.deviceDoubleMemory + i * params.hostData.dataStride;

    // Integers
    // It seems to hold that in 3 dimensions the total number of
    // bubble pairs is 10x and in two dimensions 4x numBubbles.
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avgNumNeighbors = (NUM_DIM == 3) ? 10 : 4;
    params.hostData.pairStride = avgNumNeighbors * params.hostData.dataStride;

    params.hostData.memReqI =
        sizeof(int) * (uint64_t)params.hostData.dataStride *
        ((uint64_t)DIP::PAIR1 +
         avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceIntMemory),
                           params.hostData.memReqI));

    for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
        params.dips[i] =
            params.deviceIntMemory + i * params.hostData.dataStride;

    uint32_t j = 0;
    for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
        params.dips[i] = params.dips[(uint32_t)DIP::PAIR1] +
                         avgNumNeighbors * ++j * params.hostData.dataStride;

    params.previousX.resize(params.hostData.dataStride);
    params.previousY.resize(params.hostData.dataStride);
    params.previousZ.resize(params.hostData.dataStride);

    std::cout << "Memory requirement for data:\n\tdouble: "
              << params.hostData.memReqD
              << " bytes\n\tint: " << params.hostData.memReqI
              << " bytes\n\ttotal: "
              << params.hostData.memReqI + params.hostData.memReqD << " bytes"
              << std::endl;
}

void generateStartingData(Params &params, ivec bubblesPerDim, double stdDevRad,
                          int rngSeed) {
    std::cout << "Starting to generate data for bubbles." << std::endl;
    const double avgRad = params.hostData.avgRad;

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    if (NUM_DIM == 3)
        CURAND_CALL(curandGenerateUniformDouble(generator,
                                                params.ddps[(uint32_t)DDP::Z],
                                                params.hostData.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::X], params.hostData.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::Y], params.hostData.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::RP], params.hostData.numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(
        generator, params.ddps[(uint32_t)DDP::R], params.hostData.numBubbles,
        avgRad, stdDevRad));
    CURAND_CALL(curandDestroyGenerator(generator));

    KERNEL_LAUNCH(assignDataToBubbles, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::RP],
                  params.dips[(uint32_t)DIP::INDEX], bubblesPerDim, avgRad,
                  params.hostData.numBubbles);

    params.hostConstants.averageSurfaceAreaIn =
        params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::RP],
            params.hostData.numBubbles, 0);

    params.hostData.maxBubbleRadius =
        params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Max, params.ddps[(uint32_t)DDP::R],
            params.hostData.numBubbles, 0);

    std::cout << "Updating neighbor lists." << std::endl;
    updateCellsAndNeighbors(params);

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton predictor-corrector method
    KERNEL_LAUNCH(
        resetKernel, params.defaultKernelSize, 0, 0, 0.0,
        params.hostData.numBubbles, params.ddps[(uint32_t)DDP::DXDTO],
        params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::DZDTO],
        params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::DISTANCE],
        params.ddps[(uint32_t)DDP::PATH]);

    std::cout << "Calculating some initial values as a part of setup."
              << std::endl;

    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, 0,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
        params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
        params.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(
        eulerKernel, params.pairKernelSize, 0, 0, params.hostData.numBubbles,
        params.hostData.timeStep, params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::Y],
        params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.hostData.numBubbles, params.ddps[(uint32_t)DDP::DXDTO],
                  params.ddps[(uint32_t)DDP::DYDTO],
                  params.ddps[(uint32_t)DDP::DZDTO],
                  params.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, 0,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
        params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
        params.ddps[(uint32_t)DDP::DZDTO]);
}

void initializeFromJson(const char *inputFileName, Params &params) {
    std::cout << "\n=====\nSetup\n====="
              << "Reading inputs from file \"" << inputFileName << "\""
              << std::endl;

    nlohmann::json inputJson;
    std::fstream file(inputFileName, std::ios::in);
    if (file.is_open()) {
        file >> inputJson;

        const double mu = inputJson["muZero"];
        assert(mu > 0);
        assert(inputJson["wallDragStrength"] >= 0.0 &&
               inputJson["wallDragStrength"] <= 1.0);

        params.hostData.avgRad = inputJson["avgRad"];
        params.hostConstants.minRad = 0.1 * params.hostData.avgRad;
        params.hostConstants.fZeroPerMuZero =
            (float)inputJson["sigmaZero"] * params.hostData.avgRad / mu;
        params.hostConstants.flowLbb = inputJson["flowLbb"];
        params.hostConstants.flowTfr = inputJson["flowTfr"];
        params.hostConstants.flowVel = inputJson["flowVel"];
        params.hostConstants.flowVel *= params.hostConstants.fZeroPerMuZero;
        params.hostConstants.kParameter = inputJson["kParameter"];
        params.hostConstants.kappa = inputJson["kappa"];
        params.hostConstants.skinRadius =
            (float)inputJson["skinRadius"] * params.hostData.avgRad;
        params.hostData.timeScalingFactor =
            params.hostConstants.kParameter /
            (params.hostData.avgRad * params.hostData.avgRad);
        params.hostData.errorTolerance = inputJson["errorTolerance"];
        params.hostConstants.wallDragStrength = inputJson["wallDragStrength"];
        params.hostData.snapshotFrequency = inputJson["snapshotFrequency"];
        params.hostData.minNumBubbles = inputJson["minNumBubbles"];
    } else
        throw std::runtime_error("Couldn't open input file!");

    // First calculate the size of the box and the starting number of bubbles
    dvec relDim = inputJson["boxRelDim"];
    assert(relDim.x > 0);
    assert(relDim.y > 0);
    assert(relDim.z > 0);

    relDim = relDim / relDim.x;
    const float d = 2 * params.hostData.avgRad;
    float x = (float)inputJson["numBubblesIn"] * d * d / relDim.y;
    ivec bubblesPerDim = ivec(0, 0, 0);

    if (NUM_DIM == 3) {
        x = x * d / relDim.z;
        x = std::cbrt(x);
        relDim = relDim * x;
        bubblesPerDim = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d),
                             std::ceil(relDim.z / d));
        params.hostData.numBubbles =
            bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
    } else {
        x = std::sqrt(x);
        relDim = relDim * x;
        bubblesPerDim =
            ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
        params.hostData.numBubbles = bubblesPerDim.x * bubblesPerDim.y;
    }

    params.hostConstants.tfr =
        d * bubblesPerDim.asType<double>() + params.hostConstants.lbb;
    params.hostConstants.interval =
        params.hostConstants.tfr - params.hostConstants.lbb;
    params.hostData.timeStep = inputJson["timeStepIn"];

    Constants *deviceConstants = nullptr;
    CUDA_ASSERT(cudaGetSymbolAddress(
        reinterpret_cast<void **>(&deviceConstants), dConstants));
    CUDA_CALL(cudaMemcpy(deviceConstants, &params.hostConstants,
                         sizeof(Constants), CudaMemcpyHostToDevice));

    // Reserve memory etc.
    commonSetup(params);
    generateStartingData(params, bubblesPerDim, inputJson["stdDevRad"],
                         inputJson["rngSeed"]);

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params, inputJson["numStepsToRelax"]);

    const double bubbleVolume = calculateVolumeOfBubbles(params);
    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << inputJson["phiTarget"]
              << "\nScaling the simulation box." << std::endl;

    transformPositions(params, true);

    relDim = inputJson["boxRelDim"];
    double t =
        bubbleVolume / ((float)inputJson["phiTarget"] * relDim.x * relDim.y);
    if (NUM_DIM == 3) {
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

    // Copy the updated constants to GPU
    CUDA_CALL(cudaMemcpy(deviceConstants, &params.hostConstants,
                         sizeof(Constants), CudaMemcpyHostToDevice));

    transformPositions(params, false);

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params, inputJson["numStepsToRelax"]);

    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << inputJson["phiTarget"]
              << "\n\n=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    std::cout << std::setw(10) << std::left << "#steps" << std::setw(12)
              << std::left << "dE" << std::setw(15) << std::left << "e1"
              << std::setw(15) << std::left << "e2" << std::setw(5) << std::left
              << "#searches" << std::endl;

    while (true) {
        double time = stabilize(params, inputJson["numStepsToRelax"]);
        double deltaEnergy =
            std::abs(1.0 - params.hostData.energy1 / params.hostData.energy2) /
            time;

        if (deltaEnergy < inputJson["maxDeltaEnergy"]) {
            std::cout << "Final delta energy " << deltaEnergy << " after "
                      << (numSteps + 1) * (int)inputJson["numStepsToRelax"]
                      << " steps."
                      << "\nEnergy before: " << params.hostData.energy1
                      << ", energy after: " << params.hostData.energy2
                      << ", time: " << time * params.hostData.timeScalingFactor
                      << std::endl;
            break;
        } else if (numSteps > failsafe) {
            std::cout << "Over " << failsafe * (int)inputJson["numStepsToRelax"]
                      << " steps taken and required delta energy not reached."
                      << " Check parameters." << std::endl;
            break;
        } else {
            std::cout << std::setw(10) << std::left
                      << (numSteps + 1) * (int)inputJson["numStepsToRelax"]
                      << std::setw(12) << std::left << std::setprecision(5)
                      << std::scientific << deltaEnergy << std::setw(15)
                      << std::left << std::setprecision(5) << std::fixed
                      << params.hostData.energy1 << std::setw(15) << std::left
                      << std::setprecision(5) << std::fixed
                      << params.hostData.energy2 << std::setw(5) << std::left
                      << params.hostData.numNeighborsSearched << std::endl;
            params.hostData.numNeighborsSearched = 0;
        }

        ++numSteps;
    }

    // Set starting positions
    // Avoiding batched memset, because the pointers might not be in order
    const uint64_t numBytesToCopy = sizeof(double) * params.hostData.dataStride;
    CUDA_CALL(cudaMemcpy(params.ddps[(uint32_t)DDP::X0],
                         params.ddps[(uint32_t)DDP::X], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(params.ddps[(uint32_t)DDP::Y0],
                         params.ddps[(uint32_t)DDP::Y], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(params.ddps[(uint32_t)DDP::Z0],
                         params.ddps[(uint32_t)DDP::Z], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));

    // Reset wrap counts to 0
    // Again avoiding batched memset, because the pointers might not be in order
    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_X], 0,
                         params.hostData.dataStride * sizeof(int)));

    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_Y], 0,
                         params.hostData.dataStride * sizeof(int)));

    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_Z], 0,
                         params.hostData.dataStride * sizeof(int)));

    // Reset temp for energy, and errors since integration starts after this
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.hostData.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA],
                  params.ddps[(uint32_t)DDP::ERROR]);

    // Calculate the energy at starting positions
    KERNEL_LAUNCH(
        potentialEnergyKernel, params.pairKernelSize, 0, 0,
        params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z]);

    params.hostData.energy1 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.hostData.numBubbles);
    params.hostData.timeInteger = 0;
    params.hostData.timeFraction = 0.0;
    params.hostData.timesPrinted = 1;
    params.hostData.numIntegrationSteps = 0;
}

} // namespace

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
        // Define two lambas for later use
        auto getSum = [](double *p, Params &params) -> double {
            return params.cw.reduce<double, double *, double *>(
                &cub::DeviceReduce::Sum, p, params.hostData.numBubbles);
        };

        auto getAvg = [getSum](double *p, Params &params) -> double {
            return getSum(p, params) / params.hostData.numBubbles;
        };

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
        if (params.hostData.timeInteger >= nextPrintTimeInteger &&
            params.hostData.timeFraction >= nextPrintTimeFraction) {
            // Calculate total energy
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.hostData.numBubbles,
                          params.ddps[(uint32_t)DDP::TEMP_DATA]);

            KERNEL_LAUNCH(
                potentialEnergyKernel, params.pairKernelSize, 0, 0,
                params.hostData.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                params.dips[(uint32_t)DIP::PAIR2],
                params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::TEMP_DATA],
                params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::Z]);

            params.hostData.energy2 =
                getSum(params.ddps[(uint32_t)DDP::TEMP_DATA], params);
            const double dE =
                (params.hostData.energy2 - params.hostData.energy1) /
                params.hostData.energy2;
            const double relRad =
                getAvg(params.ddps[(uint32_t)DDP::R], params) /
                params.hostData.avgRad;

            // Add values to data stream
            std::ofstream resultFile("results.dat", std::ios_base::app);
            if (resultFile.is_open()) {
                const double vx =
                    getAvg(params.ddps[(uint32_t)DDP::DXDT], params);
                const double vy =
                    getAvg(params.ddps[(uint32_t)DDP::DYDT], params);
                const double vz =
                    getAvg(params.ddps[(uint32_t)DDP::DZDT], params);
                const double vr =
                    getAvg(params.ddps[(uint32_t)DDP::DRDT], params);

                resultFile << params.hostData.timesPrinted << " " << relRad
                           << " " << params.hostData.numBubbles << " "
                           << getAvg(params.ddps[(uint32_t)DDP::PATH], params)
                           << " "
                           << getAvg(params.ddps[(uint32_t)DDP::DISTANCE],
                                     params)
                           << " " << params.hostData.energy2 << " " << dE << " "
                           << vx << " " << vy << " " << vz << " "
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
                      << params.hostData.numBubbles << std::setw(10)
                      << std::left << params.hostData.numPairs << std::setw(10)
                      << std::left << params.hostData.numStepsInTimeStep
                      << std::setw(10) << std::left
                      << params.hostData.numNeighborsSearched << std::setw(10)
                      << std::left << minTimestep << std::setw(10) << std::left
                      << maxTimestep << std::setw(10) << std::left
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
                                            params.hostData.snapshotFrequency /
                                            params.hostData.timeScalingFactor;
            const uint64_t nextSnapshotTimeInteger = (uint64_t)nextSnapshotTime;
            const double nextSnapshotTimeFraction =
                nextSnapshotTime - nextSnapshotTimeInteger;

            if (params.hostData.timeInteger >= nextSnapshotTimeInteger &&
                params.hostData.timeFraction >= nextSnapshotTimeFraction)
                saveSnapshotToFile(params);
        }

        if (resetErrors) {
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.hostData.numBubbles,
                          params.ddps[(uint32_t)DDP::ERROR]);
            resetErrors = false;
        }

        ++params.hostData.numStepsInTimeStep;
    }

    if (params.hostData.snapshotFrequency > 0.0)
        saveSnapshotToFile(params);

    deinit(params);
}
} // namespace cubble
