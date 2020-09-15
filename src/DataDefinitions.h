#pragma once

#include "CubWrapper.h"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <array>
#include <vector>

namespace {
template <typename T> void setIncr(T **p, T **prev, uint64_t stride) {
    *p = *prev;
    *prev += stride;
}
} // namespace

namespace cubble {
// Pointers to device memory holding the bubble data
struct Bubbles {
    double *x = nullptr;
    double *y = nullptr;
    double *z = nullptr;
    double *r = nullptr;

    double *xp = nullptr;
    double *yp = nullptr;
    double *zp = nullptr;
    double *rp = nullptr;

    double *dxdt = nullptr;
    double *dydt = nullptr;
    double *dzdt = nullptr;
    double *drdt = nullptr;

    double *dxdtp = nullptr;
    double *dydtp = nullptr;
    double *dzdtp = nullptr;
    double *drdtp = nullptr;

    double *dxdto = nullptr;
    double *dydto = nullptr;
    double *dzdto = nullptr;
    double *drdto = nullptr;

    double *path = nullptr;
    double *error = nullptr;
    double *temp_doubles = nullptr;
    double *temp_doubles2 = nullptr;

    double *flow_vx = nullptr;
    double *flow_vy = nullptr;
    double *flow_vz = nullptr;

    double *saved_x = nullptr;
    double *saved_y = nullptr;
    double *saved_z = nullptr;
    double *saved_r = nullptr;

    int *temp_ints = nullptr;
    int *wrap_count_x = nullptr;
    int *wrap_count_y = nullptr;
    int *wrap_count_z = nullptr;
    int *index = nullptr;
    int *num_neighbors = nullptr;

    // Count is the total number of bubbles
    int count = 0;
    // Stride is the "original" length of a row of data.
    // All the double data is saved in one big blob of memory
    // and the (double) pointers defined here are separated by
    // "sizeof(double) * stride" bytes.
    uint64_t stride = 0;

    // How many pointers of each type do we have in this struct
    const uint64_t numDP = 31;
    const uint64_t numIP = 6;

    uint64_t getMemReq() const {
        return stride * (sizeof(double) * numDP + sizeof(int) * numIP);
    }

    void *setupPointers(void *start) {
        // Point every named pointer to a separate stride of the continuous
        // memory blob
        double *prev = static_cast<double *>(start);
        setIncr(&x, &prev, stride);
        setIncr(&y, &prev, stride);
        setIncr(&z, &prev, stride);
        setIncr(&r, &prev, stride);
        setIncr(&xp, &prev, stride);
        setIncr(&yp, &prev, stride);
        setIncr(&zp, &prev, stride);
        setIncr(&rp, &prev, stride);
        setIncr(&dxdt, &prev, stride);
        setIncr(&dydt, &prev, stride);
        setIncr(&dzdt, &prev, stride);
        setIncr(&drdt, &prev, stride);
        setIncr(&dxdtp, &prev, stride);
        setIncr(&dydtp, &prev, stride);
        setIncr(&dzdtp, &prev, stride);
        setIncr(&drdtp, &prev, stride);
        setIncr(&dxdto, &prev, stride);
        setIncr(&dydto, &prev, stride);
        setIncr(&dzdto, &prev, stride);
        setIncr(&drdto, &prev, stride);
        setIncr(&path, &prev, stride);
        setIncr(&error, &prev, stride);
        setIncr(&temp_doubles, &prev, stride);
        setIncr(&temp_doubles2, &prev, stride);
        setIncr(&flow_vx, &prev, stride);
        setIncr(&flow_vy, &prev, stride);
        setIncr(&flow_vz, &prev, stride);
        setIncr(&saved_x, &prev, stride);
        setIncr(&saved_y, &prev, stride);
        setIncr(&saved_z, &prev, stride);
        setIncr(&saved_r, &prev, stride);

        int *prevI = reinterpret_cast<int *>(prev);
        setIncr(&temp_ints, &prevI, stride);
        setIncr(&wrap_count_x, &prevI, stride);
        setIncr(&wrap_count_y, &prevI, stride);
        setIncr(&wrap_count_z, &prevI, stride);
        setIncr(&index, &prevI, stride);
        setIncr(&num_neighbors, &prevI, stride);

        assert(static_cast<char *>(start) +
                   stride * (sizeof(double) * numDP + sizeof(int) * numIP) ==
               reinterpret_cast<char *>(prevI));

        return static_cast<void *>(prevI);
    }
};
static_assert(sizeof(Bubbles) % 8 == 0);

// Pointers to device memory holding the bubble pair data
struct Pairs {
    int *i = nullptr;
    int *j = nullptr;
    int *i_copy = nullptr;
    int *j_copy = nullptr;

    int count = 0;
    uint64_t stride = 0;

    uint64_t getMemReq() const { return sizeof(int) * stride * 4; }

    void *setupPointers(void *start) {
        int *prev = static_cast<int *>(start);
        setIncr(&i, &prev, stride);
        setIncr(&j, &prev, stride);
        setIncr(&i_copy, &prev, stride);
        setIncr(&j_copy, &prev, stride);

        assert(static_cast<char *>(start) + stride * sizeof(int) * 4 ==
               reinterpret_cast<char *>(prev));

        return static_cast<void *>(prev);
    }
};
static_assert(sizeof(Pairs) % 8 == 0);

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
    double bubbleVolumeMultiplier = 0.0;
};

// Only accessed by host
struct HostData {
    uint64_t numIntegrationSteps = 0;
    uint64_t numNeighborsSearched = 0;
    uint64_t numStepsInTimeStep = 0;

    uint64_t timeInteger = 0;
    double timeFraction = 0.0;
    double timeScalingFactor = 0.0;
    double timeStep = 0.0;

    double energy1 = 0.0;
    double energy2 = 0.0;

    double errorTolerance = 0.0;
    double snapshotFrequency = 0.0;
    double avgRad = 0.0;
    double maxBubbleRadius = 0.0;
    int minNumBubbles = 0;
    uint32_t numSnapshots = 0;
    uint32_t timesPrinted = 0;
};

// Store the addresses of device globals here for easy access
struct DeviceGlobalAddresses {
    void *dConstants = nullptr;
    void *dTotalArea = nullptr;
    void *dTotalOverlapArea = nullptr;
    void *dTotalOverlapAreaPerRadius = nullptr;
    void *dTotalAreaPerRadius = nullptr;
    void *dTotalVolumeNew = nullptr;
    void *dMaxError = nullptr;
    void *dMaxRadius = nullptr;
    void *dMaxExpansion = nullptr;
    void *dErrorEncountered = nullptr;
    void *dNumPairs = nullptr;
    void *dNumPairsNew = nullptr;
    void *dNumToBeDeleted = nullptr;

    void getAddresses() {
        CUDA_CALL(cudaGetSymbolAddress(&dConstants, dConstants));
        CUDA_CALL(cudaGetSymbolAddress(&dTotalArea, dTotalArea));
        CUDA_CALL(cudaGetSymbolAddress(&dTotalOverlapArea, dTotalOverlapArea));
        CUDA_CALL(cudaGetSymbolAddress(&dTotalOverlapAreaPerRadius,
                                       dTotalOverlapAreaPerRadius));
        CUDA_CALL(
            cudaGetSymbolAddress(&dTotalAreaPerRadius, dTotalAreaPerRadius));
        CUDA_CALL(cudaGetSymbolAddress(&dTotalVolumeNew, dTotalVolumeNew));
        CUDA_CALL(cudaGetSymbolAddress(&dMaxError, dMaxError));
        CUDA_CALL(cudaGetSymbolAddress(&dMaxRadius, dMaxRadius));
        CUDA_CALL(cudaGetSymbolAddress(&dMaxExpansion, dMaxExpansion));
        CUDA_CALL(cudaGetSymbolAddress(&dErrorEncountered, dErrorEncountered));
        CUDA_CALL(cudaGetSymbolAddress(&dNumPairs, dNumPairs));
        CUDA_CALL(cudaGetSymbolAddress(&dNumPairsNew, dNumPairsNew));
        CUDA_CALL(cudaGetSymbolAddress(&dNumToBeDeleted, dNumToBeDeleted));
    }
};

struct Params {
    Constants hostConstants;
    Constants *deviceConstants = nullptr;
    HostData hostData;
    DeviceGlobalAddresses addresses;

    Bubbles bubbles;
    Pairs pairs;

    CubWrapper cw;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t event1;
    cudaEvent_t event2;

    KernelSize pairKernelSize = KernelSize(dim3(1024, 1, 1), dim3(128, 1, 1));
    KernelSize defaultKernelSize;

    void *memory = nullptr;
    void *pinnedMemory = nullptr;

    std::vector<double> previousX;
    std::vector<double> previousY;
    std::vector<double> previousZ;
};

} // namespace cubble
