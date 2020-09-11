#pragma once

#include "CubWrapper.h"
#include "Util.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
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

    double *x0 = nullptr;
    double *y0 = nullptr;
    double *z0 = nullptr;

    double *path = nullptr;
    double *distance = nullptr;
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
    const uint64_t numDP = 35;
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
        setIncr(&x0, &prev, stride);
        setIncr(&y0, &prev, stride);
        setIncr(&z0, &prev, stride);
        setIncr(&path, &prev, stride);
        setIncr(&distance, &prev, stride);
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

    void associateHostPointers(void *hostPtr,
                               std::unordered_map<void *, void *> &ptrMap) {
        // Create void pointers to the allocated host memory
        // such that each pointer points to a continuous block
        // of 'stride' values. Store the doubles first, followed
        // immediately by the ints.
        std::vector<void *> hptrs;
        double *curr = static_cast<double *>(hostPtr);
        for (int i = 0; i < numDP; i++) {
            hptrs.push_back(hostPtr);
            curr += stride;
            hostPtr = static_cast<void *>(curr);
        }

        int *currI = static_cast<int *>(hostPtr);
        for (int i = 0; i < numIP; i++) {
            hptrs.push_back(hostPtr);
            currI += stride;
            hostPtr = static_cast<void *>(currI);
        }

        // Add all the device pointers defined in this struct
        // to a vector that is then sorted.
        std::vector<void *> dptrs;
        // Double pointers
        dptrs.push_back(static_cast<void *>(x));
        dptrs.push_back(static_cast<void *>(y));
        dptrs.push_back(static_cast<void *>(z));
        dptrs.push_back(static_cast<void *>(r));
        dptrs.push_back(static_cast<void *>(xp));
        dptrs.push_back(static_cast<void *>(yp));
        dptrs.push_back(static_cast<void *>(zp));
        dptrs.push_back(static_cast<void *>(rp));
        dptrs.push_back(static_cast<void *>(dxdt));
        dptrs.push_back(static_cast<void *>(dydt));
        dptrs.push_back(static_cast<void *>(dzdt));
        dptrs.push_back(static_cast<void *>(drdt));
        dptrs.push_back(static_cast<void *>(dxdtp));
        dptrs.push_back(static_cast<void *>(dydtp));
        dptrs.push_back(static_cast<void *>(dzdtp));
        dptrs.push_back(static_cast<void *>(drdtp));
        dptrs.push_back(static_cast<void *>(dxdto));
        dptrs.push_back(static_cast<void *>(dydto));
        dptrs.push_back(static_cast<void *>(dzdto));
        dptrs.push_back(static_cast<void *>(drdto));
        dptrs.push_back(static_cast<void *>(x0));
        dptrs.push_back(static_cast<void *>(y0));
        dptrs.push_back(static_cast<void *>(z0));
        dptrs.push_back(static_cast<void *>(path));
        dptrs.push_back(static_cast<void *>(distance));
        dptrs.push_back(static_cast<void *>(error));
        dptrs.push_back(static_cast<void *>(temp_doubles));
        dptrs.push_back(static_cast<void *>(temp_doubles2));
        dptrs.push_back(static_cast<void *>(flow_vx));
        dptrs.push_back(static_cast<void *>(flow_vy));
        dptrs.push_back(static_cast<void *>(flow_vz));
        dptrs.push_back(static_cast<void *>(saved_x));
        dptrs.push_back(static_cast<void *>(saved_y));
        dptrs.push_back(static_cast<void *>(saved_z));
        dptrs.push_back(static_cast<void *>(saved_r));
        // Int pointers
        dptrs.push_back(static_cast<void *>(temp_ints));
        dptrs.push_back(static_cast<void *>(wrap_count_x));
        dptrs.push_back(static_cast<void *>(wrap_count_y));
        dptrs.push_back(static_cast<void *>(wrap_count_z));
        dptrs.push_back(static_cast<void *>(index));
        dptrs.push_back(static_cast<void *>(num_neighbors));

        // Sort the doubles separate from the ints
        std::sort(dptrs.begin(), dptrs.begin() + numDP);
        std::sort(dptrs.begin() + numDP, dptrs.end());

        // Associate each device pointer with a host pointer,
        // store the pointers in the map as integers
        ptrMap.clear();
        for (int i = 0; i < dptrs.size(); i++) {
            ptrMap.insert(reinterpret_cast<intptr_t>(dptrs[i]),
                          reinterpret_cast<intptr_t>(hptrs[i]));
        }
    }

    template <typename T>
    T *getHostPtr(T *devPtr,
                  const std::unordered_map<intptr_t, intptr_t> &ptrMap) {
        return reinterpret_cast<T *>(
            ptrMap.at(reinterpret_cast<intptr_t>(devPtr)));
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

struct Params {
    Constants hostConstants;
    Constants *deviceConstants = nullptr;
    HostData hostData;

    Bubbles bubbles;
    Pairs pairs;

    CubWrapper cw;
    cudaStream_t velocityStream;
    cudaStream_t gasStream;
    cudaEvent_t event1;

    KernelSize pairKernelSize = KernelSize(dim3(1024, 1, 1), dim3(128, 1, 1));
    KernelSize defaultKernelSize;

    void *memory = nullptr;
    int *pinnedInt = nullptr;
    double *pinnedDouble = nullptr;

    std::vector<double> previousX;
    std::vector<double> previousY;
    std::vector<double> previousZ;

    int *numToBeDeleted = nullptr;
};

} // namespace cubble
