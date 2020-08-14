#include "CubWrapper.h"
#include "Kernels.cuh"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include "nlohmann/json.hpp"
#include <array>
#include <cstring>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <nvToolsExt.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace cubble {
// Device double pointer names
enum class DDP {
    X,
    Y,
    Z,
    R,

    XP,
    YP,
    ZP,
    RP,

    DXDT,
    DYDT,
    DZDT,
    DRDT,

    DXDTP,
    DYDTP,
    DZDTP,
    DRDTP,

    DXDTO,
    DYDTO,
    DZDTO,
    DRDTO,

    X0,
    Y0,
    Z0,

    PATH,
    DISTANCE,
    ERROR,
    TEMP_DATA,

    FLOW_VX,
    FLOW_VY,
    FLOW_VZ,

    SAVED_X,
    SAVED_Y,
    SAVED_Z,
    SAVED_R,

    NUM_VALUES
};

// Device int pointer names
enum class DIP {
    TEMP,

    WRAP_COUNT_X,
    WRAP_COUNT_Y,
    WRAP_COUNT_Z,

    INDEX,
    NUM_NEIGHBORS,

    PAIR1,
    PAIR2,

    PAIR1COPY,
    PAIR2COPY,

    NUM_VALUES
};

#define PRINT_PARAM(p) std::cout << #p << ": " << p << std::endl

struct SimulationState {
    dvec lbb = dvec(0.0, 0.0, 0.0);
    dvec tfr = dvec(0.0, 0.0, 0.0);
    dvec interval = dvec(0.0, 0.0, 0.0);

    uint64_t memReqD = 0;
    uint64_t memReqI = 0;
    uint64_t numIntegrationSteps = 0;
    uint64_t numNeighborsSearched = 0;
    uint64_t numStepsInTimeStep = 0;
    uint64_t timeInteger = 0;
    double timeFraction = 0.0;
    double energy1 = 0.0;
    double energy2 = 0.0;
    double maxBubbleRadius = 0.0;
    double timeStep = 0.0;
    double averageSurfaceAreaIn = 0.0;

    int numBubbles = 0;
    int maxNumCells = 0;
    int numPairs = 0;
    uint32_t numSnapshots = 0;
    uint32_t timesPrinted = 0;
    uint32_t originalDataStride = 0;
    uint32_t dataStride = 0;
    uint32_t pairStride = 0;

    bool operator==(const SimulationState &o) {
        bool equal = true;
        equal &= lbb == o.lbb;
        equal &= tfr == o.tfr;
        equal &= interval == o.interval;

        equal &= memReqD == o.memReqD;
        equal &= memReqI == o.memReqI;
        equal &= numIntegrationSteps == o.numIntegrationSteps;

        equal &= numStepsInTimeStep == o.numStepsInTimeStep;
        equal &= timeInteger == o.timeInteger;
        equal &= timeFraction == o.timeFraction;
        equal &= energy1 == o.energy1;

        equal &= energy2 == o.energy2;
        equal &= maxBubbleRadius == o.maxBubbleRadius;
        equal &= timeStep == o.timeStep;

        equal &= numBubbles == o.numBubbles;
        equal &= maxNumCells == o.maxNumCells;
        equal &= numPairs == o.numPairs;

        equal &= numSnapshots == o.numSnapshots;
        equal &= timesPrinted == o.timesPrinted;

        equal &= originalDataStride == o.originalDataStride;
        equal &= dataStride == o.dataStride;
        equal &= pairStride == o.pairStride;

        return equal;
    }

    void print() {
        std::cout << "\n---------------State---------------" << std::endl;
        PRINT_PARAM(lbb);
        PRINT_PARAM(tfr);
        PRINT_PARAM(interval);
        PRINT_PARAM(memReqD);
        PRINT_PARAM(memReqI);
        PRINT_PARAM(numIntegrationSteps);
        PRINT_PARAM(numStepsInTimeStep);
        PRINT_PARAM(timeInteger);
        PRINT_PARAM(timeFraction);
        PRINT_PARAM(energy1);
        PRINT_PARAM(energy2);
        PRINT_PARAM(maxBubbleRadius);
        PRINT_PARAM(timeStep);
        PRINT_PARAM(averageSurfaceAreaIn);
        PRINT_PARAM(numBubbles);
        PRINT_PARAM(maxNumCells);
        PRINT_PARAM(numPairs);
        PRINT_PARAM(numSnapshots);
        PRINT_PARAM(timesPrinted);
        PRINT_PARAM(originalDataStride);
        PRINT_PARAM(dataStride);
        PRINT_PARAM(pairStride);
        std::cout << "----------------End----------------\n" << std::endl;
    }
};

struct SimulationInputs {
    dvec boxRelDim = dvec(0.0, 0.0, 0.0);
    dvec flowLbb = dvec(0.0, 0.0, 0.0);
    dvec flowTfr = dvec(0.0, 0.0, 0.0);
    dvec flowVel = dvec(0.0, 0.0, 0.0);

    double avgRad = 0.0;
    double stdDevRad = 0.0;
    double minRad = 0.0;
    double phiTarget = 0.0;
    double muZero = 0.0;
    double sigmaZero = 0.0;
    double fZeroPerMuZero = 0.0;
    double errorTolerance = 0.0;
    double maxDeltaEnergy = 0.0;
    double kParameter = 0.0;
    double kappa = 0.0;
    double timeScalingFactor = 0.0;
    double timeStepIn = 0.0;
    double wallDragStrength = 0.0;
    double snapshotFrequency = 0.0;
    double skinRadius = 0.0;

    int numBubblesPerCell = 0;
    int rngSeed = 0;
    int numStepsToRelax = 0;
    int numBubblesIn = 0;
    int minNumBubbles = 0;

    bool operator==(const SimulationInputs &o) {
        bool equal = true;
        equal &= boxRelDim == o.boxRelDim;
        equal &= flowLbb == o.flowLbb;
        equal &= flowTfr == o.flowTfr;

        equal &= flowVel == o.flowVel;
        equal &= avgRad == o.avgRad;
        equal &= stdDevRad == o.stdDevRad;

        equal &= minRad == o.minRad;
        equal &= phiTarget == o.phiTarget;
        equal &= muZero == o.muZero;

        equal &= sigmaZero == o.sigmaZero;
        equal &= fZeroPerMuZero == o.fZeroPerMuZero;
        equal &= errorTolerance == o.errorTolerance;

        equal &= maxDeltaEnergy == o.maxDeltaEnergy;
        equal &= kParameter == o.kParameter;
        equal &= kappa == o.kappa;

        equal &= timeScalingFactor == o.timeScalingFactor;
        equal &= timeStepIn == o.timeStepIn;

        equal &= numBubblesPerCell == o.numBubblesPerCell;
        equal &= rngSeed == o.rngSeed;

        equal &= numStepsToRelax == o.numStepsToRelax;
        equal &= numBubblesIn == o.numBubblesIn;
        equal &= minNumBubbles == o.minNumBubbles;
        equal &= wallDragStrength == o.wallDragStrength;
        equal &= snapshotFrequency == o.snapshotFrequency;
        equal &= skinRadius == o.skinRadius;

        return equal;
    }

    void print() {
        std::cout << "\n---------------Inputs---------------" << std::endl;
        PRINT_PARAM(boxRelDim);
        PRINT_PARAM(flowLbb);
        PRINT_PARAM(flowTfr);
        PRINT_PARAM(flowVel);
        PRINT_PARAM(avgRad);
        PRINT_PARAM(stdDevRad);
        PRINT_PARAM(minRad);
        PRINT_PARAM(phiTarget);
        PRINT_PARAM(muZero);
        PRINT_PARAM(sigmaZero);
        PRINT_PARAM(fZeroPerMuZero);
        PRINT_PARAM(errorTolerance);
        PRINT_PARAM(maxDeltaEnergy);
        PRINT_PARAM(kParameter);
        PRINT_PARAM(kappa);
        PRINT_PARAM(timeScalingFactor);
        PRINT_PARAM(timeStepIn);
        PRINT_PARAM(numBubblesPerCell);
        PRINT_PARAM(rngSeed);
        PRINT_PARAM(numStepsToRelax);
        PRINT_PARAM(numBubblesIn);
        PRINT_PARAM(minNumBubbles);
        PRINT_PARAM(wallDragStrength);
        PRINT_PARAM(snapshotFrequency);
        PRINT_PARAM(skinRadius);
        std::cout << "-----------------End----------------\n" << std::endl;
    }
};

struct Params {
    SimulationState state;
    SimulationInputs inputs;
    CubWrapper cw;

    cudaStream_t velocityStream;
    cudaStream_t gasStream;

    cudaEvent_t event1;

    KernelSize pairKernelSize = KernelSize(dim3(1024, 1, 1), dim3(128, 1, 1));
    KernelSize defaultKernelSize;

    // Device memory & arrays of pointers to those memory chunks.
    int *deviceIntMemory = nullptr;
    double *deviceDoubleMemory = nullptr;
    int *pinnedInt = nullptr;
    double *pinnedDouble = nullptr;
    std::array<double *, (uint64_t)DDP::NUM_VALUES> ddps;
    std::array<int *, (uint64_t)DIP::NUM_VALUES> dips;

    std::vector<double> previousX;
    std::vector<double> previousY;
    std::vector<double> previousZ;

    int *numToBeDeleted = nullptr;
};

} // namespace cubble

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

dim3 getGridSize(Params &params) {
    const int totalNumCells = std::ceil((float)params.state.numBubbles /
                                        params.inputs.numBubblesPerCell);
    dvec relativeInterval = params.state.interval / params.state.interval.x;
    float nx = (float)totalNumCells / relativeInterval.y;
#if (NUM_DIM == 3)
    nx = std::cbrt(nx / relativeInterval.z);
#else
    nx = std::sqrt(nx);
    relativeInterval.z = 0;
#endif

    ivec grid = (nx * relativeInterval).floor() + 1;
    assert(grid.x > 0);
    assert(grid.y > 0);
    assert(grid.z > 0);

    return dim3(grid.x, grid.y, grid.z);
}

void updateCellsAndNeighbors(Params &params) {
    NVTX_RANGE_PUSH_A("Neighbors");
    params.state.numNeighborsSearched++;
    // Boundary wrap
    KERNEL_LAUNCH(wrapKernel, params.pairKernelSize, 0, 0,
                  params.state.numBubbles, params.state.lbb, params.state.tfr,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                  params.dips[(uint32_t)DIP::WRAP_COUNT_Z]);

    // Update saved values
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_X]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::X]),
        sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_Y]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::Y]),
        sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_Z]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::Z]),
        sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::SAVED_R]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::R]),
        sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));

    dim3 gridSize = getGridSize(params);
    const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

    int *offsets = params.dips[(uint32_t)DIP::PAIR1];
    int *sizes = params.dips[(uint32_t)DIP::PAIR1] + params.state.maxNumCells;
    int *cellIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 0 * params.state.dataStride;
    int *bubbleIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 1 * params.state.dataStride;
    int *sortedCellIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 2 * params.state.dataStride;
    int *sortedBubbleIndices =
        params.dips[(uint32_t)DIP::PAIR1COPY] + 3 * params.state.dataStride;

    const uint64_t resetBytes =
        sizeof(int) * params.state.pairStride *
        ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

    // Reset number of neighbors to zero as they will be calculated again
    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::NUM_NEIGHBORS], 0,
                         sizeof(int) * params.state.dataStride));

    KERNEL_LAUNCH(assignBubblesToCells, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                  params.state.lbb, params.state.tfr, cellDim,
                  params.state.numBubbles);

    params.cw.sortPairs<int, int>(
        &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices),
        sortedCellIndices, const_cast<const int *>(bubbleIndices),
        sortedBubbleIndices, params.state.numBubbles);

    params.cw.histogram<int *, int, int, int>(
        &cub::DeviceHistogram::HistogramEven, cellIndices, sizes,
        params.state.maxNumCells + 1, 0, params.state.maxNumCells,
        params.state.numBubbles);

    params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets,
                                 params.state.maxNumCells);

    auto copyAndSwap = [](Params &params, int *inds, auto &&arr, uint32_t from,
                          uint32_t to) {
        KERNEL_LAUNCH(copyKernel, params.defaultKernelSize, 0, 0,
                      params.state.numBubbles, ReorganizeType::COPY_FROM_INDEX,
                      inds, inds, arr[from], arr[to]);

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
        KERNEL_LAUNCH(neighborSearch, kernelSizeNeighbor, 0, stream, i,
                      params.state.numBubbles, params.state.maxNumCells,
                      (int)params.state.pairStride, params.inputs.skinRadius,
                      offsets, sizes, params.dips[(uint32_t)DIP::PAIR1COPY],
                      params.dips[(uint32_t)DIP::PAIR2COPY],
                      params.ddps[(uint32_t)DDP::R], params.state.interval,
                      params.ddps[(uint32_t)DDP::X],
                      params.ddps[(uint32_t)DDP::Y],
                      params.ddps[(uint32_t)DDP::Z],
                      params.dips[(uint32_t)DIP::NUM_NEIGHBORS]);
    }

    CUDA_CALL(cudaMemcpy(static_cast<void *>(&params.state.numPairs),
                         static_cast<void *>(dnp), sizeof(int),
                         cudaMemcpyDeviceToHost));

    params.cw.sortPairs<int, int>(
        &cub::DeviceRadixSort::SortPairs,
        const_cast<const int *>(params.dips[(uint32_t)DIP::PAIR1COPY]),
        params.dips[(uint32_t)DIP::PAIR1],
        const_cast<const int *>(params.dips[(uint32_t)DIP::PAIR2COPY]),
        params.dips[(uint32_t)DIP::PAIR2], params.state.numPairs);
    NVTX_RANGE_POP();
}

void deleteSmallBubbles(Params &params, int numToBeDeleted) {
    NVTX_RANGE_PUSH_A("BubbleRemoval");

    KERNEL_LAUNCH(
        swapDataCountPairs, params.pairKernelSize, 0, 0,
        params.state.numBubbles, params.inputs.minRad,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
        params.dips[(uint32_t)DIP::TEMP], params.ddps[(uint32_t)DDP::R],
        params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
        params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DXDT],
        params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DZDT],
        params.ddps[(uint32_t)DDP::DRDT], params.ddps[(uint32_t)DDP::DXDTO],
        params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::DZDTO],
        params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::X0],
        params.ddps[(uint32_t)DDP::Y0], params.ddps[(uint32_t)DDP::Z0],
        params.ddps[(uint32_t)DDP::PATH], params.ddps[(uint32_t)DDP::DISTANCE],
        params.ddps[(uint32_t)DDP::SAVED_X],
        params.ddps[(uint32_t)DDP::SAVED_Y],
        params.ddps[(uint32_t)DDP::SAVED_Z],
        params.ddps[(uint32_t)DDP::SAVED_R], params.ddps[(uint32_t)DDP::ERROR],
        params.dips[(uint32_t)DIP::WRAP_COUNT_X],
        params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
        params.dips[(uint32_t)DIP::WRAP_COUNT_Z],
        params.dips[(uint32_t)DIP::INDEX],
        params.dips[(uint32_t)DIP::NUM_NEIGHBORS]);

    KERNEL_LAUNCH(
        addVolumeFixPairs, params.pairKernelSize, 0, 0, params.state.numBubbles,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
        params.dips[(uint32_t)DIP::TEMP], params.ddps[(uint32_t)DDP::R]);

    // Update kernel sizes based on number of remaining bubbles
    params.state.numBubbles -= numToBeDeleted;
    params.defaultKernelSize = KernelSize(128, params.state.numBubbles);
    int numBlocks =
        std::min(1024, (int)std::ceil(params.state.numBubbles / 128.0));
    params.pairKernelSize = KernelSize(dim3(numBlocks, 1, 1), dim3(128, 1, 1));

    NVTX_RANGE_POP();
}

void saveSnapshotToFile(Params &params) {
    // Calculate total energy
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0,
                  params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                  params.dips[(uint32_t)DIP::PAIR2],
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA], params.state.interval,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z]);

    std::stringstream ss;
    ss << "snapshot.csv." << params.state.numSnapshots;
    std::ofstream file(ss.str().c_str(), std::ios::out);
    if (file.is_open()) {
        std::vector<double> doubleData;
        doubleData.resize(params.state.dataStride * (uint32_t)DDP::NUM_VALUES);
        for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i) {
            CUDA_CALL(cudaMemcpy(&doubleData[i * params.state.dataStride],
                                 params.ddps[i],
                                 sizeof(double) * params.state.dataStride,
                                 cudaMemcpyDeviceToHost));
        }

        std::vector<int> intData;
        intData.resize(params.state.dataStride);
        CUDA_CALL(cudaMemcpy(intData.data(), params.dips[(uint32_t)DIP::INDEX],
                             sizeof(intData[0]) * intData.size(),
                             cudaMemcpyDeviceToHost));

        if (params.state.numSnapshots == 0) {
            for (uint64_t i = 0; i < (uint64_t)params.state.numBubbles; ++i) {
                params.previousX[intData[i]] =
                    doubleData[i + 0 * params.state.dataStride];
                params.previousY[intData[i]] =
                    doubleData[i + 1 * params.state.dataStride];
                params.previousZ[intData[i]] =
                    doubleData[i + 2 * params.state.dataStride];
            }
        }

        file << "x,y,z,r,vx,vy,vz,vtot,vr,path,distance,energy,displacement,"
                "error,index\n ";
        for (uint64_t i = 0; i < (uint64_t)params.state.numBubbles; ++i) {
            const double x =
                doubleData[i + (uint32_t)DDP::X * params.state.dataStride];
            const double y =
                doubleData[i + (uint32_t)DDP::Y * params.state.dataStride];
            const double z =
                doubleData[i + (uint32_t)DDP::Z * params.state.dataStride];
            const double r =
                doubleData[i + (uint32_t)DDP::R * params.state.dataStride];
            const double vx =
                doubleData[i + (uint32_t)DDP::DXDT * params.state.dataStride];
            const double vy =
                doubleData[i + (uint32_t)DDP::DYDT * params.state.dataStride];
            const double vz =
                doubleData[i + (uint32_t)DDP::DZDT * params.state.dataStride];
            const double vr =
                doubleData[i + (uint32_t)DDP::DRDT * params.state.dataStride];
            const double path =
                doubleData[i + (uint32_t)DDP::PATH * params.state.dataStride];
            const double distance = doubleData[i + (uint32_t)DDP::DISTANCE *
                                                       params.state.dataStride];
            const double error =
                doubleData[i + (uint32_t)DDP::ERROR * params.state.dataStride];
            const double energy = doubleData[i + (uint32_t)DDP::TEMP_DATA *
                                                     params.state.dataStride];
            const double px = params.previousX[intData[i]];
            const double py = params.previousY[intData[i]];
            const double pz = params.previousZ[intData[i]];

            double displX = abs(x - px);
            displX = displX > 0.5 * params.state.interval.x
                         ? displX - params.state.interval.x
                         : displX;
            double displY = abs(y - py);
            displY = displY > 0.5 * params.state.interval.y
                         ? displY - params.state.interval.y
                         : displY;
            double displZ = abs(z - pz);
            displZ = displZ > 0.5 * params.state.interval.z
                         ? displZ - params.state.interval.z
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
            file << intData[i + 0 * params.state.dataStride];
            file << "\n";

            params.previousX[intData[i]] = x;
            params.previousY[intData[i]] = y;
            params.previousZ[intData[i]] = z;
        }

        ++params.state.numSnapshots;
    }
}

double stabilize(Params &params) {
    // This function integrates only the positions of the bubbles.
    // Gas exchange is not used. This is used for equilibrating the foam.

    double elapsedTime = 0.0;
    double error = 100000;

    // Energy before stabilization
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0,
                  params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                  params.dips[(uint32_t)DIP::PAIR2],
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA], params.state.interval,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z]);

    params.state.energy1 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.state.numBubbles);

    for (int i = 0; i < params.inputs.numStepsToRelax; ++i) {
        do {
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.state.numBubbles,
                          params.ddps[(uint32_t)DDP::DXDTP],
                          params.ddps[(uint32_t)DDP::DYDTP],
                          params.ddps[(uint32_t)DDP::DZDTP]);

            KERNEL_LAUNCH(
                predictKernel, params.pairKernelSize, 0, 0,
                params.state.numBubbles, params.state.timeStep, false,
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
                params.inputs.fZeroPerMuZero, params.dips[(uint32_t)DIP::PAIR1],
                params.dips[(uint32_t)DIP::PAIR2],
                params.ddps[(uint32_t)DDP::R], params.state.interval,
                params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::DXDTP],
                params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDTP]);

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
            KERNEL_LAUNCH(
                velocityWallKernel, params.pairKernelSize, 0,
                params.velocityStream, params.state.numBubbles,
                params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::XP],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::DXDTP],
                params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDTP], params.state.lbb,
                params.state.tfr, params.inputs.fZeroPerMuZero,
                params.inputs.wallDragStrength);
#endif
            // Correct
            KERNEL_LAUNCH(
                correctKernel, params.pairKernelSize, 0, 0,
                params.state.numBubbles, params.state.timeStep, false,
                params.inputs.minRad, params.ddps[(uint32_t)DDP::ERROR],
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
                          params.gasStream, params.state.numBubbles,
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

            if (error < params.inputs.errorTolerance &&
                params.state.timeStep < 0.1)
                params.state.timeStep *= 1.9;
            else if (error > params.inputs.errorTolerance)
                params.state.timeStep *= 0.5;

        } while (error > params.inputs.errorTolerance);

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

        elapsedTime += params.state.timeStep;

        if (2 * params.pinnedDouble[2] >= params.inputs.skinRadius) {
            updateCellsAndNeighbors(params);
        }
    }

    // Energy after stabilization
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA]);

    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0,
                  params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                  params.dips[(uint32_t)DIP::PAIR2],
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA], params.state.interval,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z]);

    params.state.energy2 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.state.numBubbles);

    return elapsedTime;
}

void velocityCalculation(Params &params) {
    // Velocity
    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, params.velocityStream,
        params.inputs.fZeroPerMuZero, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::RP],
        params.state.interval, params.ddps[(uint32_t)DDP::XP],
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
            params.state.numBubbles, params.dips[(uint32_t)DIP::NUM_NEIGHBORS],
            params.ddps[(uint32_t)DDP::DXDTP],
            params.ddps[(uint32_t)DDP::DYDTP],
            params.ddps[(uint32_t)DDP::DZDTP],
            params.ddps[(uint32_t)DDP::FLOW_VX],
            params.ddps[(uint32_t)DDP::FLOW_VY],
            params.ddps[(uint32_t)DDP::FLOW_VZ], params.ddps[(uint32_t)DDP::XP],
            params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
            params.ddps[(uint32_t)DDP::RP], params.inputs.flowVel,
            params.inputs.flowTfr, params.inputs.flowLbb);
    }
#endif

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    // Wall velocity, should be after flow so that possible drag is applied
    // correctly
    KERNEL_LAUNCH(
        velocityWallKernel, params.pairKernelSize, 0, params.velocityStream,
        params.state.numBubbles, params.ddps[(uint32_t)DDP::RP],
        params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
        params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
        params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP],
        params.state.lbb, params.state.tfr, params.inputs.fZeroPerMuZero,
        params.inputs.wallDragStrength);
#endif
}

void gasExchangeCalculation(Params &params) {
    // Gas exchange
    KERNEL_LAUNCH(
        gasExchangeKernel, params.pairKernelSize, 0, params.gasStream,
        params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.state.interval,
        params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::DRDTP],
        params.ddps[(uint32_t)DDP::TEMP_DATA], params.ddps[(uint32_t)DDP::XP],
        params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP]);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, params.pairKernelSize, 0,
                  params.gasStream, params.ddps[(uint32_t)DDP::DRDTP],
                  params.ddps[(uint32_t)DDP::RP],
                  params.ddps[(uint32_t)DDP::TEMP_DATA],
                  params.state.numBubbles, params.inputs.kappa,
                  params.inputs.kParameter, params.state.averageSurfaceAreaIn);
}

bool integrate(Params &params) {
    NVTX_RANGE_PUSH_A("Integration function");

    double error = 100000;
    uint32_t numLoopsDone = 0;

    do {
        NVTX_RANGE_PUSH_A("Integration step");

        // Reset
        KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0,
                      params.velocityStream, 0.0, params.state.numBubbles,
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
            params.state.numBubbles, params.state.timeStep, true,
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
            correctKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
            params.state.timeStep, true, params.inputs.minRad,
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
                      params.velocityStream, params.state.numBubbles,
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
            params.gasStream, params.state.numBubbles, params.state.interval,
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
        if (error < params.inputs.errorTolerance && params.state.timeStep < 0.1)
            params.state.timeStep *= 1.9;
        else if (error > params.inputs.errorTolerance)
            params.state.timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > params.inputs.errorTolerance);

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

    ++params.state.numIntegrationSteps;

    // As the total simulation time can reach very large numbers as the
    // simulation goes on it's better to keep track of the time as two separate
    // values. One large integer for the integer part and a double that is
    // <= 1.0 to which the potentially very small timeStep gets added. This
    // keeps the precision of the time relatively constant even when the
    // simulation has run a long time.
    params.state.timeFraction += params.state.timeStep;
    params.state.timeInteger += (uint64_t)params.state.timeFraction;
    params.state.timeFraction =
        params.state.timeFraction - (uint64_t)params.state.timeFraction;

    params.state.maxBubbleRadius = params.pinnedDouble[1];

    // Delete, if there are nonzero amount of bubbles with a radius
    // smaller than the minimum radius. See correctKernel for the
    // comparison & calculation.
    if (params.pinnedInt[0] > 0) {
        deleteSmallBubbles(params, params.pinnedInt[0]);
    }

    // If the boundary of the bubble with maximum sum of movement & expansion
    // has moved more than half of the "skin radius", reorder bubbles.
    // See correctKernel, comparePair for details.
    if (params.pinnedDouble[2] >= 0.5 * params.inputs.skinRadius) {
        updateCellsAndNeighbors(params);
    }

    bool continueSimulation =
        params.state.numBubbles > params.inputs.minNumBubbles;
    continueSimulation &=
        (NUM_DIM == 3)
            ? params.state.maxBubbleRadius <
                  0.5 * (params.state.tfr - params.state.lbb).getMinComponent()
            : true;

    NVTX_RANGE_POP();

    return continueSimulation;
}

void transformPositions(Params &params, bool normalize) {
    KERNEL_LAUNCH(transformPositionsKernel, params.pairKernelSize, 0, 0,
                  normalize, params.state.numBubbles, params.state.lbb,
                  params.state.tfr, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z]);
}

double calculateVolumeOfBubbles(Params &params) {
    KERNEL_LAUNCH(calculateVolumes, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA],
                  params.state.numBubbles);

    return params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.state.numBubbles);
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
    dvec temp = params.state.tfr - params.state.lbb;
    return (NUM_DIM == 3) ? temp.x * temp.y * temp.z : temp.x * temp.y;
}

#define JSON_READ(i, j, arg) i.arg = j[#arg]

void readInputs(Params &params, const char *inputFileName,
                ivec &bubblesPerDim) {
    std::cout << "Reading inputs from file \"" << inputFileName << "\""
              << std::endl;
    nlohmann::json j;
    std::fstream file(inputFileName, std::ios::in);

    if (file.is_open()) {
        file >> j;
        SimulationInputs &inputs = params.inputs;

        JSON_READ(inputs, j, phiTarget);
        JSON_READ(inputs, j, muZero);
        JSON_READ(inputs, j, sigmaZero);
        JSON_READ(inputs, j, avgRad);
        JSON_READ(inputs, j, stdDevRad);
        JSON_READ(inputs, j, errorTolerance);
        JSON_READ(inputs, j, timeStepIn);
        JSON_READ(inputs, j, rngSeed);
        JSON_READ(inputs, j, numBubblesPerCell);
        JSON_READ(inputs, j, numStepsToRelax);
        JSON_READ(inputs, j, maxDeltaEnergy);
        JSON_READ(inputs, j, kParameter);
        JSON_READ(inputs, j, numBubblesIn);
        JSON_READ(inputs, j, kappa);
        JSON_READ(inputs, j, minNumBubbles);
        JSON_READ(inputs, j, boxRelDim);
        JSON_READ(inputs, j, flowLbb);
        JSON_READ(inputs, j, flowTfr);
        JSON_READ(inputs, j, flowVel);
        JSON_READ(inputs, j, wallDragStrength);
        JSON_READ(inputs, j, snapshotFrequency);
        JSON_READ(inputs, j, skinRadius);

        assert(inputs.muZero > 0);
        assert(inputs.boxRelDim.x > 0);
        assert(inputs.boxRelDim.y > 0);
        assert(inputs.boxRelDim.z > 0);
        assert(inputs.wallDragStrength >= 0.0 &&
               inputs.wallDragStrength <= 1.0);

        inputs.fZeroPerMuZero =
            inputs.sigmaZero * inputs.avgRad / inputs.muZero;
        inputs.minRad = 0.1 * inputs.avgRad;
        inputs.timeScalingFactor =
            inputs.kParameter / (inputs.avgRad * inputs.avgRad);
        inputs.flowVel *= inputs.fZeroPerMuZero;
        inputs.skinRadius *= inputs.avgRad;
    } else
        throw std::runtime_error("Couldn't open input file!");

    // First calculate the size of the box and the starting number of bubbles
    dvec relDim = params.inputs.boxRelDim;
    relDim = relDim / relDim.x;
    const float d = 2 * params.inputs.avgRad;
    float x = params.inputs.numBubblesIn * d * d / relDim.y;
    bubblesPerDim = ivec(0, 0, 0);

    if (NUM_DIM == 3) {
        x = x * d / relDim.z;
        x = std::cbrt(x);
        relDim = relDim * x;
        bubblesPerDim = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d),
                             std::ceil(relDim.z / d));
        params.state.numBubbles =
            bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
    } else {
        x = std::sqrt(x);
        relDim = relDim * x;
        bubblesPerDim =
            ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
        params.state.numBubbles = bubblesPerDim.x * bubblesPerDim.y;
    }

    params.state.tfr = d * bubblesPerDim.asType<double>() + params.state.lbb;
    params.state.interval = params.state.tfr - params.state.lbb;
    params.state.timeStep = params.inputs.timeStepIn;

    // Determine the maximum number of Morton numbers for the simulation box
    dim3 gridDim = getGridSize(params);
    const int maxGridDim =
        gridDim.x > gridDim.y ? (gridDim.x > gridDim.z ? gridDim.x : gridDim.z)
                              : (gridDim.y > gridDim.z ? gridDim.y : gridDim.z);
    int maxNumCells = 1;
    while (maxNumCells < maxGridDim)
        maxNumCells = maxNumCells << 1;

    if (NUM_DIM == 3)
        maxNumCells = maxNumCells * maxNumCells * maxNumCells;
    else
        maxNumCells = maxNumCells * maxNumCells;

    params.state.maxNumCells = maxNumCells;

    std::cout << "Maximum (theoretical) number of cells: "
              << params.state.maxNumCells
              << ", actual grid dimensions: " << gridDim.x << ", " << gridDim.y
              << ", " << gridDim.z << std::endl;
}
#undef JSON_READ

void commonSetup(Params &params) {
    params.defaultKernelSize = KernelSize(128, params.state.numBubbles);

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
    CUDA_CALL(cudaMemcpyToSymbol(dTotalVolume, reinterpret_cast<void *>(&zero),
                                 sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(
        dVolumeMultiplier, reinterpret_cast<void *>(&zero), sizeof(double)));
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
    params.state.dataStride =
        params.state.numBubbles +
        !!(params.state.numBubbles % 32) * (32 - params.state.numBubbles % 32);

    // Doubles
    params.state.memReqD = sizeof(double) * (uint64_t)params.state.dataStride *
                           (uint64_t)DDP::NUM_VALUES;
    CUDA_ASSERT(
        cudaMalloc(reinterpret_cast<void **>(&params.deviceDoubleMemory),
                   params.state.memReqD));

    for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
        params.ddps[i] =
            params.deviceDoubleMemory + i * params.state.dataStride;

    // Integers
    // It seems to roughly hold that in 3 dimensions the total number of
    // neighbors is < (10 x numBubbles) and in 2D < (3.5 x numBubbles)
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avgNumNeighbors = (NUM_DIM == 3) ? 24 : 4;
    params.state.pairStride = avgNumNeighbors * params.state.dataStride;

    params.state.memReqI =
        sizeof(int) * (uint64_t)params.state.dataStride *
        ((uint64_t)DIP::PAIR1 +
         avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceIntMemory),
                           params.state.memReqI));

    for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
        params.dips[i] = params.deviceIntMemory + i * params.state.dataStride;

    uint32_t j = 0;
    for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
        params.dips[i] = params.dips[(uint32_t)DIP::PAIR1] +
                         avgNumNeighbors * ++j * params.state.dataStride;

    params.previousX.resize(params.state.dataStride);
    params.previousY.resize(params.state.dataStride);
    params.previousZ.resize(params.state.dataStride);

    std::cout << "Memory requirement for data:\n\tdouble: "
              << params.state.memReqD
              << " bytes\n\tint: " << params.state.memReqI
              << " bytes\n\ttotal: "
              << params.state.memReqI + params.state.memReqD << " bytes"
              << std::endl;

    params.state.print();
    params.inputs.print();
}

void generateStartingData(Params &params, ivec bubblesPerDim) {
    std::cout << "Starting to generate data for bubbles." << std::endl;
    const int rngSeed = params.inputs.rngSeed;
    const double avgRad = params.inputs.avgRad;
    const double stdDevRad = params.inputs.stdDevRad;

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    if (NUM_DIM == 3)
        CURAND_CALL(curandGenerateUniformDouble(
            generator, params.ddps[(uint32_t)DDP::Z], params.state.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::X], params.state.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::Y], params.state.numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(
        generator, params.ddps[(uint32_t)DDP::RP], params.state.numBubbles));
    CURAND_CALL(
        curandGenerateNormalDouble(generator, params.ddps[(uint32_t)DDP::R],
                                   params.state.numBubbles, avgRad, stdDevRad));
    CURAND_CALL(curandDestroyGenerator(generator));

    KERNEL_LAUNCH(assignDataToBubbles, params.pairKernelSize, 0, 0,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::XP],
                  params.ddps[(uint32_t)DDP::YP],
                  params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::RP],
                  params.dips[(uint32_t)DIP::INDEX], bubblesPerDim,
                  params.state.tfr, params.state.lbb, avgRad,
                  params.inputs.minRad, params.state.numBubbles);

    params.state.averageSurfaceAreaIn =
        params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::RP],
            params.state.numBubbles, 0);

    CUDA_CALL(cudaMemcpyAsync(
        static_cast<void *>(params.ddps[(uint32_t)DDP::RP]),
        static_cast<void *>(params.ddps[(uint32_t)DDP::R]),
        sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));

    std::cout << "Updating neighbor lists." << std::endl;

    updateCellsAndNeighbors(params);

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton predictor-corrector method
    KERNEL_LAUNCH(
        resetKernel, params.defaultKernelSize, 0, 0, 0.0,
        params.state.numBubbles, params.ddps[(uint32_t)DDP::DXDTO],
        params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::DZDTO],
        params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::DISTANCE],
        params.ddps[(uint32_t)DDP::PATH]);

    std::cout << "Calculating some initial values as a part of setup."
              << std::endl;

    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, 0,
        params.inputs.fZeroPerMuZero, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.state.interval, params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
        params.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(
        eulerKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
        params.state.timeStep, params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::Y],
        params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.state.numBubbles, params.ddps[(uint32_t)DDP::DXDTO],
                  params.ddps[(uint32_t)DDP::DYDTO],
                  params.ddps[(uint32_t)DDP::DZDTO],
                  params.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(
        velocityPairKernel, params.pairKernelSize, 0, 0,
        params.inputs.fZeroPerMuZero, params.dips[(uint32_t)DIP::PAIR1],
        params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
        params.state.interval, params.ddps[(uint32_t)DDP::X],
        params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z],
        params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
        params.ddps[(uint32_t)DDP::DZDTO]);
}

void initializeFromJson(const char *inputFileName, Params &params) {
    // Initialize everything, starting with an input .json file.
    // The end state of this function is 'prepared state' that can then be used
    // immediately to run the integration loop.

    std::cout << "\n=====\nSetup\n=====" << std::endl;
    ivec bubblesPerDim = ivec(0, 0, 0);
    readInputs(params, inputFileName, bubblesPerDim);
    commonSetup(params);

    // This parameter is used with serialization. It should be immutable and
    // never be changed after this. It just saves the original dataStride.
    params.state.originalDataStride = params.state.dataStride;

    generateStartingData(params, bubblesPerDim);

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params);

    const double bubbleVolume = calculateVolumeOfBubbles(params);

    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << params.inputs.phiTarget
              << "\nScaling the simulation box." << std::endl;

    transformPositions(params, true);

    dvec relativeSize = params.inputs.boxRelDim;
    relativeSize.z = (NUM_DIM == 2) ? 1 : relativeSize.z;
    double t = bubbleVolume / (params.inputs.phiTarget * relativeSize.x *
                               relativeSize.y * relativeSize.z);
    t = (NUM_DIM == 3) ? std::cbrt(t) : std::sqrt(t);
    params.state.tfr = dvec(t, t, t) * relativeSize;
    params.state.interval = params.state.tfr - params.state.lbb;
    params.inputs.flowTfr =
        params.state.interval * params.inputs.flowTfr + params.state.lbb;
    params.inputs.flowLbb =
        params.state.interval * params.inputs.flowLbb + params.state.lbb;

    transformPositions(params, false);

    for (uint32_t i = 0; i < 3; ++i)
        stabilize(params);

    std::cout << "Volume ratios: current: "
              << bubbleVolume / getSimulationBoxVolume(params)
              << ", target: " << params.inputs.phiTarget
              << "\n\n=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    std::cout << std::setw(10) << std::left << "#steps" << std::setw(12)
              << std::left << "dE" << std::setw(15) << std::left << "e1"
              << std::setw(15) << std::left << "e2" << std::setw(5) << std::left
              << "#searches" << std::endl;

    while (true) {
        double time = stabilize(params);
        double deltaEnergy =
            std::abs(params.state.energy2 - params.state.energy1) / time;
        deltaEnergy *= 0.5 * params.inputs.sigmaZero;

        if (deltaEnergy < params.inputs.maxDeltaEnergy) {
            std::cout << "Final delta energy " << deltaEnergy << " after "
                      << (numSteps + 1) * params.inputs.numStepsToRelax
                      << " steps."
                      << "\nEnergy before: " << params.state.energy1
                      << ", energy after: " << params.state.energy2
                      << ", time: " << time * params.inputs.timeScalingFactor
                      << std::endl;
            break;
        } else if (numSteps > failsafe) {
            std::cout << "Over " << failsafe * params.inputs.numStepsToRelax
                      << " steps taken and required delta energy not reached."
                      << " Check parameters." << std::endl;
            break;
        } else {
            std::cout << std::setw(10) << std::left
                      << (numSteps + 1) * params.inputs.numStepsToRelax
                      << std::setw(12) << std::left << std::setprecision(5)
                      << std::scientific << deltaEnergy << std::setw(15)
                      << std::left << std::setprecision(5) << std::fixed
                      << params.state.energy1 << std::setw(15) << std::left
                      << std::setprecision(5) << std::fixed
                      << params.state.energy2 << std::setw(5) << std::left
                      << params.state.numNeighborsSearched << std::endl;
            params.state.numNeighborsSearched = 0;
        }

        ++numSteps;
    }

    // Set starting positions
    // Copying one by one, because pointers get swapped and although
    // these three should be consecutive ones anyway, it's better
    // to be safe than sorry in this case.
    const uint64_t numBytesToCopy = sizeof(double) * params.state.dataStride;
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
                         params.state.dataStride * sizeof(int)));

    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_Y], 0,
                         params.state.dataStride * sizeof(int)));

    CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_Z], 0,
                         params.state.dataStride * sizeof(int)));

    // Reset temp for energy, and errors since integration starts after this
    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                  params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::TEMP_DATA],
                  params.ddps[(uint32_t)DDP::ERROR]);

    // Calculate the energy at starting positions
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0,
                  params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                  params.dips[(uint32_t)DIP::PAIR2],
                  params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP_DATA], params.state.interval,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::Z]);

    params.state.energy1 = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP_DATA],
        params.state.numBubbles);
    params.state.timeInteger = 0;
    params.state.timeFraction = 0.0;
    params.state.timesPrinted = 1;
    params.state.numIntegrationSteps = 0;
}

// Needed for serialization and deserialization
#define NAME_MAX_LEN 64
#define HEADER_SIZE 256 // <-- Needs to be > 2x NAME_MAX_LEN

void initializeFromBinary(const char *inputFileName, Params &params) {
    std::cout << "Initializing simulation from a binary dump." << std::endl;

    std::ifstream inFile(inputFileName, std::ifstream::binary);
    if (inFile.is_open()) {
        inFile.seekg(0, inFile.end);
        const uint64_t fileSize = inFile.tellg();
        inFile.seekg(0);
        uint64_t fs =
            sizeof(params.state) + sizeof(params.inputs) + HEADER_SIZE;
        if (fileSize <= fs) {
            std::stringstream ss;
            ss << "The given binary file is incorrect size. Check "
                  "that the file is correct. File size: "
               << fileSize << ", required minimum: " << fs;
            throw std::runtime_error(ss.str());
        }

        std::vector<char> byteData;
        byteData.resize(fileSize);
        inFile.read(byteData.data(), fileSize);

        uint64_t offset = 0;
        std::array<char, HEADER_SIZE> header;

        // Header
        std::memcpy(static_cast<void *>(header.data()),
                    static_cast<void *>(&byteData[offset]), header.size());
        offset += header.size();

        // Check some values from header for compatibility
        uint32_t headerOffset = 0;
        char *nulLoc =
            std::find(header.begin() + headerOffset,
                      header.begin() + headerOffset + NAME_MAX_LEN, '\0');
        std::string hostName(header.begin() + headerOffset, nulLoc);
        headerOffset += NAME_MAX_LEN;

        nulLoc = std::find(header.begin() + headerOffset,
                           header.begin() + headerOffset + NAME_MAX_LEN, '\0');
        std::string gpuName(header.begin() + headerOffset, nulLoc);
        headerOffset += NAME_MAX_LEN;

        int binNumDim = 0;
        std::memcpy(static_cast<void *>(&binNumDim),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        int binUseProfiling = 0;
        std::memcpy(static_cast<void *>(&binUseProfiling),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        int binUseFlow = 0;
        std::memcpy(static_cast<void *>(&binUseFlow),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        int binPbcx = 0;
        std::memcpy(static_cast<void *>(&binPbcx),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        int binPbcy = 0;
        std::memcpy(static_cast<void *>(&binPbcy),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        int binPbcz = 0;
        std::memcpy(static_cast<void *>(&binPbcz),
                    static_cast<void *>(&header[headerOffset]), sizeof(int));
        headerOffset += sizeof(int);

        std::cout << "Binary header:\n\tHostname: " << hostName
                  << "\n\tGPU name: " << gpuName << "\n\tNUM_DIM: " << binNumDim
                  << "\n\tUSE_PROFILING: " << binUseProfiling
                  << "\n\tUSE_FLOW: " << binUseFlow << "\n\tPBC_X: " << binPbcx
                  << "\n\tPBC_Y: " << binPbcy << "\n\tPBC_Z: " << binPbcz
                  << std::endl;

        // Get current host name
        std::array<char, NAME_MAX_LEN> charArr;
        gethostname(charArr.data(), charArr.size());
        nulLoc = std::find(charArr.data(), charArr.end(), '\0');
        hostName = std::string(charArr.begin(), nulLoc);

        // Get current GPU name
        cudaDeviceProp prop;
        int device = 0;
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaGetDevice(&device));
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));
        gpuName = std::string(prop.name);
        gpuName = gpuName.substr(0, NAME_MAX_LEN);

        std::cout << "Current program:\n\tHostname: " << hostName
                  << "\n\tGPU name: " << gpuName << "\n\tNUM_DIM: " << NUM_DIM
                  << "\n\tUSE_PROFILING: " << USE_PROFILING
                  << "\n\tUSE_FLOW: " << USE_FLOW << "\n\tPBC_X: " << PBC_X
                  << "\n\tPBC_Y: " << PBC_Y << "\n\tPBC_Z: " << PBC_Z
                  << std::endl;

        const bool isBinaryCompatible =
            (NUM_DIM == binNumDim) && (USE_PROFILING == binUseProfiling) &&
            (USE_FLOW == binUseFlow) && (PBC_X == binPbcx) &&
            (PBC_Y == binPbcy) && (PBC_Z == binPbcz);
        if (!isBinaryCompatible)
            throw std::runtime_error("Incompatible binary file!");

        // State
        std::memcpy(static_cast<void *>(&params.state),
                    static_cast<void *>(&byteData[offset]),
                    sizeof(params.state));
        offset += sizeof(params.state);

        // Inputs
        std::memcpy(static_cast<void *>(&params.inputs),
                    static_cast<void *>(&byteData[offset]),
                    sizeof(params.inputs));
        offset += sizeof(params.inputs);

        int *dnp = nullptr;
        CUDA_ASSERT(
            cudaGetSymbolAddress(reinterpret_cast<void **>(&dnp), dNumPairs));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dnp),
                             static_cast<void *>(&params.state.numPairs),
                             sizeof(int), cudaMemcpyHostToDevice));

        // This function reserves memory & sets up the device pointers.
        commonSetup(params);

        const uint64_t doubleBytes = sizeof(double) *
                                     (uint32_t)DDP::NUM_VALUES / 2 *
                                     params.state.dataStride;
        const uint64_t intBytes = sizeof(int) * 4 * params.state.dataStride;

        // Previous positions of bubbles are saved on the cpu. They are accessed
        // with the immutable index of a bubble and thus they 'must' contain all
        // the data from the start of the simulation. This could be done in a
        // better way (by e.g. having a separate map where immutable indices are
        // translated to indices to these vectors) but I've thought that's just
        // unnecessary complication.
        params.previousX.resize(params.state.originalDataStride);
        params.previousY.resize(params.state.originalDataStride);
        params.previousZ.resize(params.state.originalDataStride);
        const uint64_t previousPosBytes =
            params.previousX.size() * sizeof(params.previousX[0]);

        fs = sizeof(params.state) + sizeof(params.inputs) + doubleBytes +
             intBytes + header.size() + 3 * previousPosBytes;
        if (fileSize != fs) {
            std::stringstream ss;
            ss << "The given binary file is incorrect size. Check "
                  "that the file is correct. File size: "
               << fileSize << ", required minimum: " << fs;
            throw std::runtime_error(ss.str());
        }

        // Doubles
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceDoubleMemory),
                             static_cast<void *>(&byteData[offset]),
                             doubleBytes, cudaMemcpyHostToDevice));
        offset += doubleBytes;

        // Previous positions
        std::memcpy(static_cast<void *>(params.previousX.data()),
                    static_cast<void *>(&byteData[offset]), previousPosBytes);
        offset += previousPosBytes;

        std::memcpy(static_cast<void *>(params.previousY.data()),
                    static_cast<void *>(&byteData[offset]), previousPosBytes);
        offset += previousPosBytes;

        std::memcpy(static_cast<void *>(params.previousZ.data()),
                    static_cast<void *>(&byteData[offset]), previousPosBytes);
        offset += previousPosBytes;

        // Ints
        CUDA_CALL(cudaMemcpy(
            static_cast<void *>(params.dips[(uint32_t)DIP::WRAP_COUNT_X]),
            static_cast<void *>(&byteData[offset]), intBytes,
            cudaMemcpyHostToDevice));
        offset += intBytes;

        // All the data should be used at this point
        if (offset != byteData.size()) {
            std::stringstream ss;
            ss << "The given binary file is incorrect size. Check "
                  "that the file is correct. Used data: "
               << offset << ", total size: " << byteData.size();
            throw std::runtime_error(ss.str());
        }

        // Setup pairs. Unnecessary to serialize them.
        updateCellsAndNeighbors(params);
    } else
        throw std::runtime_error("Couldn't open binary file for reading!");

    std::cout << "Binary initialization done." << std::endl;
}

void serializeStateAndData(const char *outputFileName, Params &params) {
    std::cout << "Serializing simulation state and data to a binary file."
              << std::endl;

    std::ofstream outFile(outputFileName, std::ofstream::binary);
    if (outFile.is_open()) {
        // Get host name
        std::array<char, NAME_MAX_LEN> charArr;
        gethostname(charArr.data(), charArr.size());
        char *nulLoc = std::find(charArr.data(), charArr.end(), '\0');
        std::string hostName(charArr.begin(), nulLoc);

        // Get GPU name
        cudaDeviceProp prop;
        int device = 0;
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaGetDevice(&device));
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));
        std::string gpuName(prop.name);
        gpuName = gpuName.substr(0, NAME_MAX_LEN);

        std::array<char, HEADER_SIZE> header;
        uint64_t offset = 0;

        strncpy(&header[offset], hostName.data(), NAME_MAX_LEN);
        offset += NAME_MAX_LEN;

        strncpy(&header[offset], gpuName.data(), NAME_MAX_LEN);
        offset += NAME_MAX_LEN;

        int temp = NUM_DIM;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        temp = USE_PROFILING;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        temp = USE_FLOW;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        temp = PBC_X;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        temp = PBC_Y;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        temp = PBC_Z;
        std::memcpy(static_cast<void *>(&header[offset]),
                    static_cast<void *>(&temp), sizeof(int));
        offset += sizeof(int);

        // Divisible by 32 and >= numBubbles
        const uint64_t newDataStride =
            params.state.numBubbles + !!(params.state.numBubbles % 32) *
                                          (32 - params.state.numBubbles % 32);
        const uint32_t numIntComponents = 4;
        const uint32_t numDoubleComponents = (uint32_t)DDP::NUM_VALUES / 2;
        const uint64_t doubleDataSize =
            numDoubleComponents * newDataStride * sizeof(double);
        const uint64_t intDataSize =
            numIntComponents * newDataStride * sizeof(int);
        const uint64_t totalSize =
            sizeof(params.state) + sizeof(params.inputs) + doubleDataSize +
            intDataSize + header.size() +
            3 * params.previousX.size() * sizeof(params.previousX[0]);

        std::vector<char> byteData;
        byteData.resize(totalSize);
        offset = 0;

        // Header
        std::memcpy(static_cast<void *>(&byteData[offset]),
                    static_cast<void *>(header.data()), header.size());
        offset += header.size();

        // State
        std::memcpy(static_cast<void *>(&byteData[offset]),
                    static_cast<void *>(&params.state), sizeof(params.state));
        offset += sizeof(params.state);

        // Inputs
        std::memcpy(static_cast<void *>(&byteData[offset]),
                    static_cast<void *>(&params.inputs), sizeof(params.inputs));
        offset += sizeof(params.inputs);

        // Doubles
        for (uint32_t i = 0; i < numDoubleComponents; ++i) {
            const uint64_t bytesToCopy = sizeof(double) * newDataStride;
            CUDA_CALL(cudaMemcpy(static_cast<void *>(&byteData[offset]),
                                 static_cast<void *>(params.ddps[i]),
                                 bytesToCopy, cudaMemcpyDeviceToHost));

            offset += bytesToCopy;
        }

        // CPU vectors for previous bubble positions
        {
            const uint64_t bytesToCopy =
                params.previousX.size() * sizeof(params.previousX[0]);

            std::memcpy(static_cast<void *>(&byteData[offset]),
                        static_cast<void *>(params.previousX.data()),
                        bytesToCopy);
            offset += bytesToCopy;

            std::memcpy(static_cast<void *>(&byteData[offset]),
                        static_cast<void *>(params.previousY.data()),
                        bytesToCopy);
            offset += bytesToCopy;

            std::memcpy(static_cast<void *>(&byteData[offset]),
                        static_cast<void *>(params.previousZ.data()),
                        bytesToCopy);
            offset += bytesToCopy;
        }

        // Ints
        // Copy line by line, since the pointers in dips
        // might not be in any particular order
        for (uint32_t i = 0; i < numIntComponents; ++i) {
            const uint64_t bytesToCopy = sizeof(int) * newDataStride;
            CUDA_CALL(
                cudaMemcpy(static_cast<void *>(&byteData[offset]),
                           static_cast<void *>(
                               params.dips[(uint32_t)DIP::WRAP_COUNT_X + i]),
                           bytesToCopy, cudaMemcpyDeviceToHost));

            offset += bytesToCopy;
        }

        if (offset != totalSize || offset != byteData.size())
            throw std::runtime_error(
                "Error in data calculation at serialization!");

        std::cout << "Writing data. Calculated size: " << totalSize
                  << "\nbyte vector size: " << byteData.size()
                  << "\noffset: " << offset
                  << "\ndouble data: " << doubleDataSize
                  << "\nint data: " << intDataSize
                  << "\nheader size: " << header.size() << "\nprev pos size: "
                  << 3 * params.previousX.size() * sizeof(params.previousX[0])
                  << "\nstate size: " << sizeof(params.state)
                  << "\ninputs size: " << sizeof(params.inputs) << std::endl;

        outFile.write(byteData.data(), byteData.size());
        outFile.close();
    } else
        throw std::runtime_error("Couldn't open file for writing!");

    std::cout << "Serialization done." << std::endl;
}

} // namespace

namespace cubble {
sig_atomic_t signalReceived = 0;

void signalHandler(int sigNum) {
    signalReceived = (sigNum == SIGUSR1) ? 1 : signalReceived;
}

void run(std::string &&inputFileName, std::string &&outputFileName) {
    // Register signal handler
    signal(SIGUSR1, signalHandler);

    Params params;
    // If input file name ends with .bin, it's a serialized state.
    if (inputFileName.compare(inputFileName.size() - 4, 4, ".bin") == 0)
        initializeFromBinary(inputFileName.c_str(), params);
    else {
        initializeFromJson(inputFileName.c_str(), params);
        if (params.inputs.snapshotFrequency > 0.0)
            saveSnapshotToFile(params);
    }

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

    while (continueIntegration) {
        continueIntegration = integrate(params);

        // For profiling
        CUDA_PROFILER_START(params.state.numIntegrationSteps == 2000);
        CUDA_PROFILER_STOP(params.state.numIntegrationSteps == 5000,
                           continueIntegration);

        continueIntegration &= signalReceived == 0;

        // Track timestep
        minTimestep = params.state.timeStep < minTimestep
                          ? params.state.timeStep
                          : minTimestep;
        maxTimestep = params.state.timeStep > maxTimestep
                          ? params.state.timeStep
                          : maxTimestep;
        avgTimestep += params.state.timeStep;

        // Here we compare potentially very large integers (> 10e6) to each
        // other and small doubles (<= 1.0) to each other to preserve precision.
        const double nextPrintTime =
            params.state.timesPrinted / params.inputs.timeScalingFactor;
        const uint64_t nextPrintTimeInteger = (uint64_t)nextPrintTime;
        const double nextPrintTimeFraction =
            nextPrintTime - nextPrintTimeInteger;

        // Print at the earliest possible moment when simulation time is larger
        // than scaled time
        if (params.state.timeInteger >= nextPrintTimeInteger &&
            params.state.timeFraction >= nextPrintTimeFraction) {
            // Calculate total energy
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.state.numBubbles,
                          params.ddps[(uint32_t)DDP::TEMP_DATA]);

            KERNEL_LAUNCH(
                potentialEnergyKernel, params.pairKernelSize, 0, 0,
                params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1],
                params.dips[(uint32_t)DIP::PAIR2],
                params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::TEMP_DATA], params.state.interval,
                params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::Z]);

            auto getSum = [](double *p, Params &params) -> double {
                return params.cw.reduce<double, double *, double *>(
                    &cub::DeviceReduce::Sum, p, params.state.numBubbles);
            };

            auto getAvg = [getSum](double *p, Params &params) -> double {
                return getSum(p, params) / params.state.numBubbles;
            };

            params.state.energy2 =
                getSum(params.ddps[(uint32_t)DDP::TEMP_DATA], params);
            const double dE = (params.state.energy2 - params.state.energy1) /
                              params.state.energy2;
            const double relRad =
                getAvg(params.ddps[(uint32_t)DDP::R], params) /
                params.inputs.avgRad;

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

                resultFile << params.state.timesPrinted << " " << relRad << " "
                           << params.state.numBubbles << " "
                           << getAvg(params.ddps[(uint32_t)DDP::PATH], params)
                           << " "
                           << getAvg(params.ddps[(uint32_t)DDP::DISTANCE],
                                     params)
                           << " " << params.state.energy2 << " " << dE << " "
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
            std::cout << std::setw(10) << std::left << params.state.timesPrinted
                      << std::setw(10) << std::left << std::setprecision(6)
                      << std::fixed << phi << std::setw(10) << std::left
                      << std::setprecision(6) << std::fixed << relRad
                      << std::setw(10) << std::left << params.state.numBubbles
                      << std::setw(10) << std::left << params.state.numPairs
                      << std::setw(10) << std::left
                      << params.state.numStepsInTimeStep << std::setw(10)
                      << std::left << params.state.numNeighborsSearched
                      << std::setw(10) << std::left << minTimestep
                      << std::setw(10) << std::left << maxTimestep
                      << std::setw(10) << std::left
                      << avgTimestep / params.state.numStepsInTimeStep
                      << std::endl;

            ++params.state.timesPrinted;
            params.state.numStepsInTimeStep = 0;
            params.state.energy1 = params.state.energy2;
            params.state.numNeighborsSearched = 0;
            minTimestep = 9999999.9;
            maxTimestep = -1.0;
            avgTimestep = 0.0;
            resetErrors = true;
        }

        // Save snapshot
        if (params.inputs.snapshotFrequency > 0.0) {
            const double nextSnapshotTime = params.state.numSnapshots /
                                            params.inputs.snapshotFrequency /
                                            params.inputs.timeScalingFactor;
            const uint64_t nextSnapshotTimeInteger = (uint64_t)nextSnapshotTime;
            const double nextSnapshotTimeFraction =
                nextSnapshotTime - nextSnapshotTimeInteger;

            if (params.state.timeInteger >= nextSnapshotTimeInteger &&
                params.state.timeFraction >= nextSnapshotTimeFraction)
                saveSnapshotToFile(params);
        }

        if (resetErrors) {
            KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0,
                          params.state.numBubbles,
                          params.ddps[(uint32_t)DDP::ERROR]);
            resetErrors = false;
        }

        ++params.state.numStepsInTimeStep;
    }

    if (signalReceived == 1) {
        std::cout << "Timeout signal received." << std::endl;
        serializeStateAndData(outputFileName.c_str(), params);
    } else if (params.inputs.snapshotFrequency > 0.0)
        saveSnapshotToFile(params);

    deinit(params);
}
} // namespace cubble
