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
    double *path_new = nullptr;
    double *error = nullptr;

    double *flow_vx = nullptr;
    double *flow_vy = nullptr;
    double *flow_vz = nullptr;

    double *saved_x = nullptr;
    double *saved_y = nullptr;
    double *saved_z = nullptr;
    double *saved_r = nullptr;

    int32_t *wrap_count_x = nullptr;
    int32_t *wrap_count_y = nullptr;
    int32_t *wrap_count_z = nullptr;
    int32_t *index = nullptr;
    int32_t *num_neighbors = nullptr;

    // Count is the total number of bubbles
    int32_t count = 0;
    // Stride is the "original" length of a row of data.
    // All the double data is saved in one big blob of memory
    // and the (double) pointers defined here are separated by
    // "sizeof(double) * stride" bytes.
    uint64_t stride = 0;

    // How many pointers of each type do we have in this struct
    const uint64_t num_dp = 30;
    const uint64_t num_ip = 5;

    uint64_t getMemReq() const {
        return stride * (sizeof(double) * num_dp + sizeof(int32_t) * num_ip);
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
        setIncr(&path_new, &prev, stride);
        setIncr(&error, &prev, stride);
        setIncr(&flow_vx, &prev, stride);
        setIncr(&flow_vy, &prev, stride);
        setIncr(&flow_vz, &prev, stride);
        setIncr(&saved_x, &prev, stride);
        setIncr(&saved_y, &prev, stride);
        setIncr(&saved_z, &prev, stride);
        setIncr(&saved_r, &prev, stride);

        int32_t *prev_i = reinterpret_cast<int32_t *>(prev);
        setIncr(&wrap_count_x, &prev_i, stride);
        setIncr(&wrap_count_y, &prev_i, stride);
        setIncr(&wrap_count_z, &prev_i, stride);
        setIncr(&index, &prev_i, stride);
        setIncr(&num_neighbors, &prev_i, stride);

        assert(static_cast<char *>(start) +
                   stride *
                       (sizeof(double) * numDP + sizeof(int32_t) * numIP) ==
               reinterpret_cast<char *>(prevI));

        return static_cast<void *>(prev_i);
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
    dvec flow_lbb = dvec(0.0, 0.0, 0.0);
    dvec flow_tfr = dvec(0.0, 0.0, 0.0);
    dvec flow_vel = dvec(0.0, 0.0, 0.0);

    double average_surface_area_in = 0.0;
    double min_rad = 0.0;
    double f_zero_per_mu_zero = 0.0;
    double k_parameter = 0.0;
    double kappa = 0.0;
    double wall_drag_strength = 0.0;
    double skin_radius = 0.3;
    double bubble_volume_multiplier = 0.0;

    int32_t dimensionality = 0;

    bool x_wall = false;
    bool y_wall = false;
    bool z_wall = false;

    void print() {
        printf("\tlower back bottom: (%g, %g, %g)", lbb.x, lbb.y, lbb.z);
        printf("\n\ttop front right: (%g, %g, %g)", tfr.x, tfr.y, tfr.z);
        printf("\n\tinterval: (%g, %g, %g)", interval.x, interval.y,
               interval.z);
        printf("\n\tflow lbb: (%g, %g, %g)", flow_lbb.x, flow_lbb.y,
               flow_lbb.z);
        printf("\n\tflow tfr: (%g, %g, %g)", flow_tfr.x, flow_tfr.y,
               flow_tfr.z);
        printf("\n\tflow vel: (%g, %g, %g)", flow_vel.x, flow_vel.y,
               flow_vel.z);
        printf("\n\tminimum radius: %g", min_rad);
        printf("\n\tf0/mu0: %g", f_zero_per_mu_zero);
        printf("\n\tk parameter: %g", k_parameter);
        printf("\n\tkappa: %g", kappa);
        printf("\n\twall drag: %g", wall_drag_strength);
        printf("\n\tskin radius: %g", skin_radius);
        printf("\n\tdimensions: %d", dimensionality);
        printf("\n\tx has wall: %d", x_wall);
        printf("\n\ty has wall: %d", y_wall);
        printf("\n\tz has wall: %d\n", z_wall);
    }
};

struct IntegrationParams {
    bool use_gas_exchange = false;
    bool use_flow = false;
    bool increment_path = false;
    bool error_too_large = true;
    double max_radius = 0.0;
    double max_expansion = 0.0;
    double max_error = 0.0;
    int32_t *h_num_to_be_deleted = nullptr;
};

// Only accessed by host
struct HostData {
    uint64_t num_integration_steps = 0;
    uint64_t num_neighbors_searched = 0;
    uint64_t num_steps_in_time_step = 0;

    uint64_t time_integer = 0;
    double time_fraction = 0.0;
    double time_scaling_factor = 0.0;
    double time_step = 0.0001;

    double energy1 = 0.0;
    double energy2 = 0.0;

    double error_tolerance = 0.0;
    double snapshot_frequency = 0.0;
    double avg_rad = 0.0;
    double max_bubble_radius = 0.0;
    int32_t min_num_bubbles = 0;
    uint32_t times_printed = 0;
    uint32_t num_snapshots = 0;

    bool add_flow = false;

    void print() {
        printf("\terror tolerance: %g", error_tolerance);
        printf("\n\tsnapshot frequency: %g", snapshot_frequency);
        printf("\n\taverage radius: %g", avg_rad);
        printf("\n\tminimum number of bubbles: %d", min_num_bubbles);
        printf("\n\timpose flow: %d\n", add_flow);
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
    int32_t *wrap_count_x = nullptr;
    int32_t *wrap_count_y = nullptr;
    int32_t *wrap_count_z = nullptr;

    dvec interval = {};

    // Starting positions
    std::vector<double> x0 = {};
    std::vector<double> y0 = {};
    std::vector<double> z0 = {};

    std::string name = {};

    cudaEvent_t event = {};
};

struct Params {
    Constants host_constants = {};
    Constants *device_constants = nullptr;
    HostData host_data = {};
    SnapshotParams snapshot_params = {};

    Bubbles bubbles = {};
    Pairs pairs = {};

    std::thread io_thread = {};

    dim3 block_grid = dim3(GRID_SIZE, 1, 1);
    dim3 thread_block = dim3(BLOCK_SIZE, 1, 1);

    void *memory = nullptr;
    void *pinned_memory = nullptr;

    double *temp_d1 = nullptr;
    double *block_max = nullptr;
    int32_t *temp_i = nullptr;
    int32_t *temp_pair1 = nullptr;
    int32_t *temp_pair2 = nullptr;

    std::vector<double> previous_x = {};
    std::vector<double> previous_y = {};
    std::vector<double> previous_z = {};
    std::vector<uint8_t> host_memory = {};
    std::vector<double> maximums = {};

    void setTempPointers(void *ptr) {
        temp_pair1 = static_cast<int32_t *>(ptr);
        temp_pair2 = temp_pair1 + pairs.stride;
        temp_i = temp_pair2 + pairs.stride;
        temp_d1 = reinterpret_cast<double *>(temp_i + bubbles.stride);
        block_max = temp_d1 + bubbles.stride;
    }

    uint64_t getTempMemReq() const {
        return (2 * pairs.stride + bubbles.stride) * sizeof(int32_t) +
               (bubbles.stride + 3 * GRID_SIZE) * sizeof(double);
    }
};

} // namespace cubble
