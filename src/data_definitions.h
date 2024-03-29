/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "vec.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <string>
#include <thread>
#include <vector>

#define BLOCK_SIZE 384
#define GRID_SIZE 5120

// TODO: use aosoa
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
    double *pathNew = nullptr;
    double *error = nullptr;

    double *flowVx = nullptr;
    double *flowVy = nullptr;
    double *flowVz = nullptr;

    double *savedX = nullptr;
    double *savedY = nullptr;
    double *savedZ = nullptr;
    double *savedR = nullptr;

    int32_t *wrapCountX = nullptr;
    int32_t *wrapCountY = nullptr;
    int32_t *wrapCountZ = nullptr;
    int32_t *index = nullptr;
    int32_t *numNeighbors = nullptr;

    // Count is the total number of bubbles
    int32_t count = 0;
    // Stride is the "original" length of a row of data.
    // All the double data is saved in one big blob of memory
    // and the (double) pointers defined here are separated by
    // "sizeof(double) * stride" bytes.
    uint64_t stride = 0;

    // How many pointers of each type do we have in this struct
    const uint64_t numDP = 30;
    const uint64_t numIP = 5;

    uint64_t getMemReq() const {
        return stride * (sizeof(double) * numDP + sizeof(int32_t) * numIP);
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
        setIncr(&pathNew, &prev, stride);
        setIncr(&error, &prev, stride);
        setIncr(&flowVx, &prev, stride);
        setIncr(&flowVy, &prev, stride);
        setIncr(&flowVz, &prev, stride);
        setIncr(&savedX, &prev, stride);
        setIncr(&savedY, &prev, stride);
        setIncr(&savedZ, &prev, stride);
        setIncr(&savedR, &prev, stride);

        int32_t *prevI = reinterpret_cast<int32_t *>(prev);
        setIncr(&wrapCountX, &prevI, stride);
        setIncr(&wrapCountY, &prevI, stride);
        setIncr(&wrapCountZ, &prevI, stride);
        setIncr(&index, &prevI, stride);
        setIncr(&numNeighbors, &prevI, stride);

        assert(static_cast<char *>(start) +
                   stride *
                       (sizeof(double) * numDP + sizeof(int32_t) * numIP) ==
               reinterpret_cast<char *>(prevI));

        return static_cast<void *>(prevI);
    }

    void print() { printf("\t#bubbles: %i, stride: %li\n", count, stride); }
};
static_assert(sizeof(Bubbles) % 8 == 0);

// Pointers to device memory holding the bubble pair data
struct Pairs {
    int32_t *i = nullptr;
    int32_t *j = nullptr;

    int32_t count = 0;
    uint64_t stride = 0;

    uint64_t getMemReq() const { return sizeof(int32_t) * stride * 2; }

    void *setupPointers(void *start) {
        int32_t *prev = static_cast<int32_t *>(start);
        setIncr(&i, &prev, stride);
        setIncr(&j, &prev, stride);

        assert(static_cast<char *>(start) + stride * sizeof(int32_t) * 2 ==
               reinterpret_cast<char *>(prev));

        return static_cast<void *>(prev);
    }

    void print() { printf("\t#pairs: %d, stride: %li\n", count, stride); }
};
static_assert(sizeof(Pairs) % 8 == 0);

// These values never change after init
// TODO: const
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
    double skinRadius = 0.3;
    double bubbleVolumeMultiplier = 0.0;

    int32_t dimensionality = 0;

    bool xWall = false;
    bool yWall = false;
    bool zWall = false;

    void print() {
        printf("\tlower back bottom: (%g, %g, %g)", lbb.x, lbb.y, lbb.z);
        printf("\n\ttop front right: (%g, %g, %g)", tfr.x, tfr.y, tfr.z);
        printf("\n\tinterval: (%g, %g, %g)", interval.x, interval.y,
               interval.z);
        printf("\n\tflow lbb: (%g, %g, %g)", flowLbb.x, flowLbb.y, flowLbb.z);
        printf("\n\tflow tfr: (%g, %g, %g)", flowTfr.x, flowTfr.y, flowTfr.z);
        printf("\n\tflow vel: (%g, %g, %g)", flowVel.x, flowVel.y, flowVel.z);
        printf("\n\tminimum radius: %g", minRad);
        printf("\n\tf0/mu0: %g", fZeroPerMuZero);
        printf("\n\tk parameter: %g", kParameter);
        printf("\n\tkappa: %g", kappa);
        printf("\n\twall drag: %g", wallDragStrength);
        printf("\n\tskin radius: %g", skinRadius);
        printf("\n\tdimensions: %d", dimensionality);
        printf("\n\tx has wall: %d", xWall);
        printf("\n\ty has wall: %d", yWall);
        printf("\n\tz has wall: %d\n", zWall);
    }
};

struct IntegrationParams {
    bool useGasExchange = false;
    bool useFlow = false;
    bool incrementPath = false;
    bool errorTooLarge = true;
    double maxRadius = 0.0;
    double maxExpansion = 0.0;
    double maxError = 0.0;
    int32_t *hNumToBeDeleted = nullptr;
};

// Only accessed by host
struct HostData {
    uint64_t numIntegrationSteps = 0;
    uint64_t numNeighborsSearched = 0;
    uint64_t numStepsInTimeStep = 0;

    uint64_t timeInteger = 0;
    double timeFraction = 0.0;
    double timeScalingFactor = 0.0;
    double timeStep = 0.0001;

    double energy1 = 0.0;
    double energy2 = 0.0;

    double errorTolerance = 0.0;
    double snapshotFrequency = 0.0;
    double avgRad = 0.0;
    double maxBubbleRadius = 0.0;
    int32_t minNumBubbles = 0;
    uint32_t timesPrinted = 0;
    uint32_t numSnapshots = 0;

    bool addFlow = false;

    void print() {
        printf("\terror tolerance: %g", errorTolerance);
        printf("\n\tsnapshot frequency: %g", snapshotFrequency);
        printf("\n\taverage radius: %g", avgRad);
        printf("\n\tminimum number of bubbles: %d", minNumBubbles);
        printf("\n\timpose flow: %d\n", addFlow);
    }
};

struct SnapshotParams {
    uint64_t count = 0;

    double *x = nullptr;
    double *y = nullptr;
    double *z = nullptr;
    double *r = nullptr;
    double *vx = nullptr;
    double *vy = nullptr;
    double *vz = nullptr;
    double *vr = nullptr;
    double *path = nullptr;
    double *error = nullptr;
    double *energy = nullptr;
    int32_t *index = nullptr;
    int32_t *wrapCountX = nullptr;
    int32_t *wrapCountY = nullptr;
    int32_t *wrapCountZ = nullptr;

    dvec interval = {};

    // Starting positions
    std::vector<double> x0 = {};
    std::vector<double> y0 = {};
    std::vector<double> z0 = {};

    std::string name = {};

    cudaEvent_t event = {};
};

struct Params {
    Constants hostConstants = {};
    Constants *deviceConstants = nullptr;
    HostData hostData = {};
    SnapshotParams snapshotParams = {};

    Bubbles bubbles = {};
    Pairs pairs = {};

    std::thread ioThread = {};

    dim3 blockGrid = dim3(GRID_SIZE, 1, 1);
    dim3 threadBlock = dim3(BLOCK_SIZE, 1, 1);

    void *memory = nullptr;
    void *pinnedMemory = nullptr;

    double *tempD1 = nullptr;
    double *blockMax = nullptr;
    int32_t *tempI = nullptr;
    int32_t *tempPair1 = nullptr;
    int32_t *tempPair2 = nullptr;

    std::vector<double> previousX = {};
    std::vector<double> previousY = {};
    std::vector<double> previousZ = {};
    std::vector<uint8_t> hostMemory = {};
    std::vector<double> maximums = {};

    void setTempPointers(void *ptr) {
        tempPair1 = static_cast<int32_t *>(ptr);
        tempPair2 = tempPair1 + pairs.stride;
        tempI = tempPair2 + pairs.stride;
        tempD1 = reinterpret_cast<double *>(tempI + bubbles.stride);
        blockMax = tempD1 + bubbles.stride;
    }

    uint64_t getTempMemReq() const {
        return (2 * pairs.stride + bubbles.stride) * sizeof(int32_t) +
               (bubbles.stride + 3 * GRID_SIZE) * sizeof(double);
    }
};

} // namespace cubble
