#pragma once

#include "CubWrapper.h"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <array>
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

// Pretty much read-only memory by both, host and device
// after an initial setup
struct Constants {
    dvec lbb = dvec(0.0, 0.0, 0.0);
    dvec tfr = dvec(0.0, 0.0, 0.0);
    dvec interval = dvec(0.0, 0.0, 0.0);
    dvec flowLbb = dvec(0.0, 0.0, 0.0);
    dvec flowTfr = dvec(0.0, 0.0, 0.0);
    dvec flowVel = dvec(0.0, 0.0, 0.0);

    double averageSurfaceAreaIn = 0.0;
    double minRad = 0.0;
    double fZeroPerMuZero = 0.0;
    double kParameter = 0.0;
    double kappa = 0.0;
    double wallDragStrength = 0.0;
    double skinRadius = 0.0;
};

// Only accessed by host
struct HostData {
    uint64_t memReqD = 0;
    uint64_t memReqI = 0;
    uint64_t numIntegrationSteps = 0;
    uint64_t numNeighborsSearched = 0;
    uint64_t numStepsInTimeStep = 0;
    uint64_t timeInteger = 0;
    double timeFraction = 0.0;
    double energy1 = 0.0;
    double energy2 = 0.0;
    double timeScalingFactor = 0.0;
    double errorTolerance = 0.0;
    double snapshotFrequency = 0.0;
    double avgRad = 0.0;
    double timeStep = 0.0;
    double maxBubbleRadius = 0.0;

    int numBubbles = 0;
    int numPairs = 0;
    int minNumBubbles = 0;
    uint32_t numSnapshots = 0;
    uint32_t timesPrinted = 0;
    uint32_t dataStride = 0;
    uint32_t pairStride = 0;
};

struct Params {
    Constants hostConstants;
    HostData hostData;

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
