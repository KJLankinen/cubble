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

#include "data_definitions.h"
#include "device_globals.cuh"
#include "kernels.cuh"
#include "macros.h"
#include "particle_box.h"
#include "util.cuh"
#include "vec.h"

#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <exception>
#include <fstream>
#include <nvToolsExt.h>
#include <sstream>
#include <stdio.h>
#include <string>

namespace {
using namespace cubble;

double totalEnergy(Params &params) {
    nvtxRangePush("Energy");
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.temp_d1);
    KERNEL_LAUNCH(potentialEnergy, params, 0, 0, params.bubbles, params.pairs,
                  params.temp_d1);

    void *cub_ptr = static_cast<void *>(params.temp_pair2);
    double total = 0.0;
    void *cub_output = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&cub_output, d_max_radius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cub_ptr, params.pairs.getMemReq() / 2,
               params.temp_d1, static_cast<double *>(cub_output),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total), d_max_radius,
                                   sizeof(double)));
    nvtxRangePop();

    return total;
}

void searchNeighbors(Params &params) {
    nvtxRangePush("Neighbors");
    params.host_data.num_neighbors_searched++;

    KERNEL_LAUNCH(wrapOverPeriodicBoundaries, params, 0, 0, params.bubbles);

    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.pairs.i), 0,
                              params.pairs.getMemReq(), 0));

    // Minimum size of cell is twice the sum of the skin and max bubble radius
    ivec cell_dim = ivec(floor(params.host_constants.interval /
                               (2 * (params.host_data.max_bubble_radius +
                                     params.host_constants.skin_radius))));
    cell_dim.z = cell_dim.z > 0 ? cell_dim.z : 1;
    const int32_t num_cells = cell_dim.x * cell_dim.y * cell_dim.z;

    // Note that these pointers alias memory, that is used by this function.
    // Don't fiddle with these, unless you know what you're doing.
    int32_t *cell_offsets = params.pairs.i;
    int32_t *cell_sizes = cell_offsets + num_cells;
    int32_t *cell_indices = cell_sizes + num_cells;
    int32_t *bubble_indices = cell_indices + params.bubbles.stride;
    int32_t *histogram = bubble_indices + params.bubbles.stride;
    void *cub_ptr = static_cast<void *>(params.pairs.j);
    uint64_t max_cub_mem = params.pairs.getMemReq() / 2;

    KERNEL_LAUNCH(cellByPosition, params, 0, 0, cell_indices, cell_sizes,
                  cell_dim, params.bubbles);

    CUB_LAUNCH(&cub::DeviceScan::InclusiveSum, cub_ptr, max_cub_mem, cell_sizes,
               cell_offsets, num_cells, (cudaStream_t)0, false);

    KERNEL_LAUNCH(indexByCell, params, 0, 0, cell_indices, cell_offsets,
                  bubble_indices, params.bubbles.count);

    {
        KERNEL_LAUNCH(reorganizeByIndex, params, 0, 0, params.bubbles,
                      const_cast<const int32_t *>(bubble_indices));
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
        params.bubbles.dxdto = params.bubbles.flow_vx;
        params.bubbles.flow_vx = swapper;

        int32_t *swapper_i = params.bubbles.index;
        params.bubbles.index = params.bubbles.wrap_count_z;
        params.bubbles.wrap_count_z = params.bubbles.wrap_count_y;
        params.bubbles.wrap_count_y = params.bubbles.wrap_count_x;
        params.bubbles.wrap_count_x = params.bubbles.num_neighbors;
        params.bubbles.num_neighbors = swapper_i;
    }

    int32_t zero = 0;
    CUDA_CALL(cudaMemcpyToSymbol(d_num_pairs, static_cast<void *>(&zero),
                                 sizeof(int32_t)));

    int32_t num_cells_to_search = 5;
    if (params.host_constants.dimensionality == 3) {
        num_cells_to_search = 14;
    }

    KERNEL_LAUNCH(neighborSearch, params, 0, 0, num_cells, num_cells_to_search,
                  cell_dim, cell_offsets, cell_sizes, histogram,
                  params.temp_pair1, params.temp_pair2, params.bubbles);

    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&params.pairs.count),
                                   d_num_pairs, sizeof(int32_t)));

    CUB_LAUNCH(&cub::DeviceScan::InclusiveSum, cub_ptr, max_cub_mem, histogram,
               params.bubbles.num_neighbors, params.bubbles.count,
               (cudaStream_t)0, false);

    KERNEL_LAUNCH(sortPairs, params, 0, 0, params.bubbles, params.pairs,
                  params.temp_pair1, params.temp_pair2);

    CUDA_CALL(cudaMemset(static_cast<void *>(params.bubbles.num_neighbors), 0,
                         params.bubbles.count * sizeof(int32_t)));

    KERNEL_LAUNCH(countNumNeighbors, params, 0, 0, params.bubbles,
                  params.pairs);
    nvtxRangePop();
}

void removeBubbles(Params &params, int32_t numToBeDeleted) {
    nvtxRangePush("Removal");
    KERNEL_LAUNCH(swapDataCountPairs, params, 0, 0, params.bubbles,
                  params.pairs, params.temp_i);

    KERNEL_LAUNCH(addVolumeFixPairs, params, 0, 0, params.bubbles, params.pairs,
                  params.temp_i);

    params.bubbles.count -= numToBeDeleted;
    nvtxRangePop();
}

void step(Params &params, IntegrationParams &ip) {
    nvtxRangePush("Integration step");

    double &ts = params.host_data.time_step;

    KERNEL_LAUNCH(initGlobals, params, 0, 0);
    KERNEL_LAUNCH(preIntegrate, params, 0, 0, ts, ip.use_gas_exchange,
                  params.bubbles, params.temp_d1);
    const uint32_t dyn_shared_mem_bytes =
        (params.host_constants.dimensionality + ip.use_gas_exchange * 4 +
         ip.use_flow * params.host_constants.dimensionality) *
        BLOCK_SIZE * sizeof(double);
    KERNEL_LAUNCH(pairwiseInteraction, params, dyn_shared_mem_bytes, 0,
                  params.bubbles, params.pairs, params.temp_d1,
                  ip.use_gas_exchange, ip.use_flow);
    KERNEL_LAUNCH(postIntegrate, params, 0, 0, ts, ip.use_gas_exchange,
                  ip.increment_path, ip.use_flow, params.bubbles,
                  params.block_max, params.temp_d1, params.temp_i);

    if (ip.use_gas_exchange) {
        assert(nullptr != ip.hNumToBeDeleted && "Given pointer is nullptr");
        // Copy numToBeDeleted
        CUDA_CALL(cudaMemcpyFromSymbolAsync(
            static_cast<void *>(ip.h_num_to_be_deleted), d_num_to_be_deleted,
            sizeof(int32_t), 0, cudaMemcpyDefault, 0));
    }

    void *mem_start = static_cast<void *>(params.maximums.data());
    CUDA_CALL(cudaMemcpy(mem_start, static_cast<void *>(params.block_max),
                         3 * GRID_SIZE * sizeof(double), cudaMemcpyDefault));

    ip.max_radius = 0.0;
    ip.max_expansion = 0.0;
    ip.max_error = 0.0;
    uint32_t n = 1;
    if (params.bubbles.count > BLOCK_SIZE) {
        float temp = static_cast<float>(params.bubbles.count) / BLOCK_SIZE;
        n = static_cast<uint32_t>(std::ceil(temp));
        n = std::min(n, static_cast<uint32_t>(GRID_SIZE));
    }
    double *p = static_cast<double *>(mem_start);
    for (uint32_t i = 0; i < n; i++) {
        ip.max_error = fmax(ip.max_error, p[i]);
        ip.max_radius = fmax(ip.max_radius, p[i + GRID_SIZE]);
        ip.max_expansion = fmax(ip.max_expansion, p[i + 2 * GRID_SIZE]);
    }

    ip.error_too_large = ip.max_error > params.host_data.error_tolerance;
    const bool increase_ts =
        ip.max_error < 0.45 * params.host_data.error_tolerance && ts < 10;
    if (ip.error_too_large) {
        ts *= 0.37;
    } else if (increase_ts) {
        ts *= 1.269;
    }

    nvtxRangePop();
}

void integrate(Params &params, IntegrationParams &ip) {
    nvtxRangePush("Intergration");

    do {
        step(params, ip);
    } while (ip.error_too_large);

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

    swapper = params.bubbles.x;
    params.bubbles.x = params.bubbles.xp;
    params.bubbles.xp = swapper;

    swapper = params.bubbles.y;
    params.bubbles.y = params.bubbles.yp;
    params.bubbles.yp = swapper;

    swapper = params.bubbles.z;
    params.bubbles.z = params.bubbles.zp;
    params.bubbles.zp = swapper;

    if (ip.use_gas_exchange) {
        swapper = params.bubbles.r;
        params.bubbles.r = params.bubbles.rp;
        params.bubbles.rp = swapper;

        swapper = params.bubbles.drdto;
        params.bubbles.drdto = params.bubbles.drdt;
        params.bubbles.drdt = params.bubbles.drdtp;
        params.bubbles.drdtp = swapper;
    }

    if (ip.increment_path) {
        swapper = params.bubbles.path;
        params.bubbles.path = params.bubbles.path_new;
        params.bubbles.path_new = swapper;
    }

    ++params.host_data.num_integration_steps;

    // As the total simulation time can reach very large numbers as the
    // simulation goes on it's better to keep track of the time as two separate
    // values. One large integer for the integer part and a double that is
    // <= 1.0 to which the potentially very small timeStep gets added. This
    // keeps the precision of the time relatively constant even when the
    // simulation has run a long time.
    params.host_data.time_fraction += params.host_data.time_step;
    params.host_data.time_integer += (uint64_t)params.host_data.time_fraction;
    params.host_data.time_fraction = params.host_data.time_fraction -
                                     (uint64_t)params.host_data.time_fraction;

    if (ip.use_gas_exchange) {
        params.host_data.max_bubble_radius = ip.max_radius;
        if (*(ip.h_num_to_be_deleted) > 0) {
            removeBubbles(params, *(ip.h_num_to_be_deleted));
        }
    }

    if (ip.max_expansion >= 0.5 * params.host_constants.skin_radius) {
        searchNeighbors(params);
        if (false == ip.use_gas_exchange) {
            // After searchNeighbors r is correct,
            // but rp is trash. pairwiseInteraction always uses
            // predicted values, so copy r to rp
            uint64_t bytes = params.bubbles.stride * sizeof(double);
            CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                                      static_cast<void *>(params.bubbles.r),
                                      bytes, cudaMemcpyDefault, 0));
        }
    }

    nvtxRangePop();
}

void stabilize(Params &params, int32_t numStepsToRelax) {
    nvtxRangePush("Stabilization");
    params.host_data.energy1 = totalEnergy(params);

    IntegrationParams ip;
    ip.use_gas_exchange = false;
    ip.use_flow = false;
    ip.increment_path = false;
    ip.error_too_large = true;
    ip.max_radius = 0.0;
    ip.max_expansion = 0.0;
    ip.max_error = 0.0;

    nvtxRangePush("For-loop");
    for (int32_t i = 0; i < numStepsToRelax; ++i) {
        integrate(params, ip);
    }
    nvtxRangePop();

    params.host_data.energy2 = totalEnergy(params);
    nvtxRangePop();
}

double totalVolume(Params &params) {
    nvtxRangePush("Volume");
    KERNEL_LAUNCH(calculateVolumes, params, 0, 0, params.bubbles,
                  params.temp_d1);

    void *cub_ptr = static_cast<void *>(params.temp_pair2);
    double total = 0.0;
    void *cub_output = nullptr;
    CUDA_CALL(cudaGetSymbolAddress(&cub_output, d_max_radius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cub_ptr, params.pairs.getMemReq() / 2,
               params.temp_d1, static_cast<double *>(cub_output),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total), d_max_radius,
                                   sizeof(double)));
    nvtxRangePop();

    return total;
}

double boxVolume(Params &params) {
    dvec temp = params.host_constants.interval;
    return (params.host_constants.dimensionality == 3)
               ? temp.x * temp.y * temp.z
               : temp.x * temp.y;
}

void saveSnapshot(Params &params) {
    // Calculate energies of bubbles to tempD1, but don't reduce.
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.temp_d1);
    KERNEL_LAUNCH(potentialEnergy, params, 0, 0, params.bubbles, params.pairs,
                  params.temp_d1);

    // Make sure the thread is not working
    if (params.io_thread.joinable()) {
        params.io_thread.join();
    }

    // Copy all device memory to host.
    void *mem_start = static_cast<void *>(params.host_memory.data());
    CUDA_CALL(cudaMemcpyAsync(mem_start, params.memory,
                              params.host_memory.size() *
                                  sizeof(params.host_memory[0]),
                              cudaMemcpyDefault, 0));
    CUDA_CALL(cudaEventRecord(params.snapshot_params.event, 0));

    // This lambda helps calculate the host address for each pointer from the
    // device address.
    auto get_host_ptr = [&params, &mem_start](auto devPtr) -> decltype(devPtr) {
        return static_cast<decltype(devPtr)>(mem_start) +
               (devPtr - static_cast<decltype(devPtr)>(params.memory));
    };

    // Get the host pointers from the device pointers
    params.snapshot_params.x = get_host_ptr(params.bubbles.x);
    params.snapshot_params.y = get_host_ptr(params.bubbles.y);
    params.snapshot_params.z = get_host_ptr(params.bubbles.z);
    params.snapshot_params.r = get_host_ptr(params.bubbles.r);
    params.snapshot_params.vx = get_host_ptr(params.bubbles.dxdt);
    params.snapshot_params.vy = get_host_ptr(params.bubbles.dydt);
    params.snapshot_params.vz = get_host_ptr(params.bubbles.dzdt);
    params.snapshot_params.vr = get_host_ptr(params.bubbles.drdt);
    params.snapshot_params.path = get_host_ptr(params.bubbles.path);
    params.snapshot_params.error = get_host_ptr(params.bubbles.error);
    params.snapshot_params.energy = get_host_ptr(params.temp_d1);
    params.snapshot_params.index = get_host_ptr(params.bubbles.index);
    params.snapshot_params.wrap_count_x =
        get_host_ptr(params.bubbles.wrap_count_x);
    params.snapshot_params.wrap_count_y =
        get_host_ptr(params.bubbles.wrap_count_y);
    params.snapshot_params.wrap_count_z =
        get_host_ptr(params.bubbles.wrap_count_z);

    params.snapshot_params.count = params.bubbles.count;

    // TODO: add distance from start
    auto write_snapshot = [](const SnapshotParams &snapshotParams,
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
                const int32_t ind = snapshotParams.index[i];
                const double xi = snapshotParams.x[i];
                const double yi = snapshotParams.y[i];
                const double zi = snapshotParams.z[i];
                const double vxi = snapshotParams.vx[i];
                const double vyi = snapshotParams.vy[i];
                const double vzi = snapshotParams.vz[i];
                const double px = xPrev[ind];
                const double py = yPrev[ind];
                const double pz = zPrev[ind];

                double displ_x = abs(xi - px);
                displ_x = displ_x > 0.5 * snapshotParams.interval.x
                              ? displ_x - snapshotParams.interval.x
                              : displ_x;
                double displ_y = abs(yi - py);
                displ_y = displ_y > 0.5 * snapshotParams.interval.y
                              ? displ_y - snapshotParams.interval.y
                              : displ_y;
                double displ_z = abs(zi - pz);
                displ_z = displ_z > 0.5 * snapshotParams.interval.z
                              ? displ_z - snapshotParams.interval.z
                              : displ_z;

                file << xi - 0.5 * snapshotParams.interval.x;
                file << ",";
                file << yi - 0.5 * snapshotParams.interval.y;
                file << ",";
                file << zi - 0.5 * snapshotParams.interval.z;
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
                file << sqrt(displ_x * displ_x + displ_y * displ_y +
                             displ_z * displ_z);
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
    params.io_thread =
        std::thread(write_snapshot, std::cref(params.snapshot_params),
                    params.host_data.num_snapshots++, params.previous_x.data(),
                    params.previous_y.data(), params.previous_z.data());
}

void checkSnapshot(Params &params) {
    if (params.host_data.snapshot_frequency > 0.0) {
        const double next_snapshot_time =
            params.host_data.num_snapshots /
            (params.host_data.snapshot_frequency *
             params.host_data.time_scaling_factor);
        const uint64_t next_snapshot_time_integer =
            (uint64_t)next_snapshot_time;
        const double next_snapshot_time_fraction =
            next_snapshot_time - next_snapshot_time_integer;

        const bool is_snapshot_time =
            params.host_data.time_integer > next_snapshot_time_integer ||
            (params.host_data.time_integer == next_snapshot_time_integer &&
             params.host_data.time_fraction >= next_snapshot_time_fraction);

        if (is_snapshot_time) {
            saveSnapshot(params);
        }
    }
}

void end(Params &params, double min_interval) {
    if (params.bubbles.count <= params.host_data.min_num_bubbles) {
        printf("Stopping simulation, since the number of bubbles left in the "
               "simulation (%d) is less than or equal to the specified minimum "
               "(%d)\n",
               params.bubbles.count, params.host_data.min_num_bubbles);
    } else if (params.host_data.max_bubble_radius > min_interval) {
        dvec temp = params.host_constants.interval;
        printf("Stopping simulation, since the radius of the largest bubble "
               "(%g) is greater than the simulation box (%g, %g, %g)\n",
               params.host_data.max_bubble_radius, temp.x, temp.y, temp.z);
    } else {
        printf("Stopping simulation for an unknown reason...\n");
    }

    if (params.host_data.snapshot_frequency > 0.0) {
        saveSnapshot(params);
    }

    printf("Cleaning up...\n");
    if (params.io_thread.joinable()) {
        params.io_thread.join();
    }

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(params.device_constants)));
    CUDA_CALL(cudaFree(params.memory));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinned_memory)));
}

bool checkEndConditions(Params &params, double min_interval) {
    return params.bubbles.count > params.host_data.min_num_bubbles &&
           params.host_data.max_bubble_radius < min_interval;
}

void printRelevantInfoOfCurrentDevice() {
    cudaDeviceProp prop;
    int32_t device = 0;

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));

    printf("\n---------------Properties of current device----------------");
    printf("\n\tGeneral\n\t-------");
    printf("\n\tName: %s", prop.name);
    printf("\n\tCompute capability: %d.%d", prop.major, prop.minor);
    printf("\n\n\tMemory\n\t------");
    printf("\n\tGlobal: %lu B", prop.totalGlobalMem);
    printf("\n\tShared per block: %lu B", prop.sharedMemPerBlock);
    printf("\n\tConstant: %lu B", prop.totalConstMem);
    printf("\n\tRegisters per block: %d", prop.regsPerBlock);
    printf("\n\n\tWarp, threads, blocks, grid\n\t---------------------------");
    printf("\n\tWarp size: %d", prop.warpSize);
    printf("\n\tThreads per block: %d", prop.maxThreadsPerBlock);
    printf("\n\tBlock size: (%d, %d, %d)", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\n\tGrid size: (%d, %d, %d)", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n\tMultiprocessor count: %d\n", prop.multiProcessorCount);
    printf("\nIf you want more info, see %s:%d and"
           "\n'Device Management' section of the CUDA Runtime API docs."
           "\n---------------------------------------------------------\n",
           __FILE__, __LINE__);
}

void printProgress(Params &params, double &min_timestep, double &max_timestep,
                   double &avg_timestep, bool &reset_errors, bool first) {
    if (first) {
        printf("\n===========\nIntegration\n===========\n");
        printf("%-5s ", "T");
        printf("%-8s ", "phi");
        printf("%-6s ", "R");
        printf("%-11s ", "dE");
        printf("%9s ", "#b   ");
        printf("%10s ", "#pairs");
        printf("%-6s ", "#steps");
        printf("%-9s ", "#searches");
        printf("%-11s ", "min ts");
        printf("%-11s ", "max ts");
        printf("%-11s \n", "avg ts");

    } else {
        const double &e1 = params.host_data.energy1;
        const double &e2 = params.host_data.energy2;

        const double next_print_time = params.host_data.times_printed /
                                       params.host_data.time_scaling_factor;
        const uint64_t next_print_time_integer = (uint64_t)next_print_time;
        const double next_print_time_fraction =
            next_print_time - next_print_time_integer;
        const bool print =
            params.host_data.time_integer > next_print_time_integer ||
            (params.host_data.time_integer == next_print_time_integer &&
             params.host_data.time_fraction >= next_print_time_fraction);
        if (print) {
            // Define lambda for calculating averages of some values
            auto get_avg = [&params](double *p, Bubbles &bubbles) -> double {
                void *cub_ptr = static_cast<void *>(params.temp_pair2);
                void *cub_output = nullptr;
                double total = 0.0;
                CUDA_CALL(cudaGetSymbolAddress(&cub_output, d_max_radius));
                CUB_LAUNCH(&cub::DeviceReduce::Sum, cub_ptr,
                           params.pairs.getMemReq() / 2, p,
                           static_cast<double *>(cub_output),
                           params.bubbles.count, (cudaStream_t)0, false);
                CUDA_CALL(cudaMemcpyFromSymbol(static_cast<void *>(&total),
                                               d_max_radius, sizeof(double)));

                return total / bubbles.count;
            };

            params.host_data.energy2 = totalEnergy(params);

            double de = std::abs(e2 - e1);
            if (de > 0.0) {
                de *= 2.0 / (e2 + e1);
            }
            const double rel_rad = get_avg(params.bubbles.r, params.bubbles) /
                                   params.host_data.avg_rad;

            // Add values to data stream
            std::ofstream result_file("results.dat", std::ios_base::app);
            if (result_file.is_open()) {
                const double vx = get_avg(params.bubbles.dxdt, params.bubbles);
                const double vy = get_avg(params.bubbles.dydt, params.bubbles);
                const double vz = get_avg(params.bubbles.dzdt, params.bubbles);
                const double vr = get_avg(params.bubbles.drdt, params.bubbles);

                result_file << params.host_data.times_printed << " " << rel_rad
                            << " " << params.bubbles.count << " "
                            << get_avg(params.bubbles.path, params.bubbles)
                            << " " << params.host_data.energy2 << " " << de
                            << " " << vx << " " << vy << " " << vz << " "
                            << sqrt(vx * vx + vy * vy + vz * vz) << " " << vr
                            << "\n";
            } else {
                printf("Couldn't open file stream to append results to!\n");
            }

            const double phi = totalVolume(params) / boxVolume(params);

            printf("%-5d ", params.host_data.times_printed);
            printf("%-#8.6g ", phi);
            printf("%-#6.4g ", rel_rad);
            printf("%-9.5e ", de);
            printf("%9d ", params.bubbles.count);
            printf("%10d ", params.pairs.count);
            printf("%6ld ", params.host_data.num_steps_in_time_step);
            printf("%-9ld ", params.host_data.num_neighbors_searched);
            printf("%-9.5e ", min_timestep);
            printf("%-9.5e ", max_timestep);
            printf("%-9.5e \n",
                   avg_timestep / params.host_data.num_steps_in_time_step);

            ++params.host_data.times_printed;
            params.host_data.num_steps_in_time_step = 0;
            params.host_data.energy1 = params.host_data.energy2;
            params.host_data.num_neighbors_searched = 0;

            min_timestep = 9999999.9;
            max_timestep = -1.0;
            avg_timestep = 0.0;
            reset_errors = true;
        }
    }
}

void trackTimeStep(double &ts, double &min_timestep, double &max_timestep,
                   double &avg_timestep) {
    min_timestep = ts < min_timestep ? ts : min_timestep;
    max_timestep = ts > max_timestep ? ts : max_timestep;
    avg_timestep += ts;
}

nlohmann::json parse(const char *inputFileName) {
    printf("Reading inputs from %s\n", inputFileName);
    nlohmann::json input_json;
    std::fstream file(inputFileName, std::ios::in);
    if (file.is_open()) {
        file >> input_json;
    } else {
        throw std::runtime_error("Couldn't open input file!");
    }

    return input_json;
}

void initializeHostConstants(Params &params, const nlohmann::json &input_json) {
    const auto constants = input_json["constants"];
    const auto bubbles = input_json["bubbles"];
    const auto box = input_json["box"];
    const auto wall = box["wall"];
    const auto flow = input_json["flow"];

    params.host_constants.skin_radius *= (double)bubbles["radius"]["mean"];
    params.host_constants.min_rad = 0.1 * (double)bubbles["radius"]["mean"];
    params.host_constants.f_zero_per_mu_zero =
        (double)constants["sigma"]["value"] *
        (double)bubbles["radius"]["mean"] / (double)constants["mu"]["value"];
    params.host_constants.k_parameter = constants["K"]["value"];
    params.host_constants.kappa = constants["kappa"]["value"];
    params.host_constants.flow_lbb = flow["lbb"];
    params.host_constants.flow_tfr = flow["tfr"];
    params.host_constants.flow_vel = flow["velocity"];
    params.host_constants.flow_vel *= params.host_constants.f_zero_per_mu_zero;
    params.host_constants.wall_drag_strength = wall["drag"];
    params.host_constants.x_wall = 1 == wall["x"];
    params.host_constants.y_wall = 1 == wall["y"];
    params.host_constants.z_wall = 1 == wall["z"];
    params.host_constants.dimensionality = box["dimensionality"];
}

void initializeHostData(Params &params, const nlohmann::json &input_json) {
    const auto bubbles = input_json["bubbles"];
    const auto flow = input_json["flow"];

    params.host_data.avg_rad = bubbles["radius"]["mean"];
    params.host_data.min_num_bubbles = bubbles["numEnd"];
    params.host_data.time_scaling_factor =
        params.host_constants.k_parameter /
        (params.host_data.avg_rad * params.host_data.avg_rad);
    params.host_data.add_flow = 1 == flow["impose"];
    params.host_data.error_tolerance = input_json["errorTolerance"]["value"];
    params.host_data.snapshot_frequency = input_json["snapShot"]["frequency"];

    params.snapshot_params.name = input_json["snapShot"]["filename"];
}

void computeSizeOfBox(Params &params, const nlohmann::json &input_json) {
    const auto bubbles = input_json["bubbles"];
    const auto box = input_json["box"];

    // Calculate the size of the box and the starting number of bubbles
    const float d = 2 * params.host_data.avg_rad;
    const ParticleBox particle_box(box["relativeDimensions"],
                                   bubbles["numStart"], box["dimensionality"]);
    params.bubbles.count = particle_box.num_particles;

    params.host_constants.tfr = d * dvec(particle_box.particles_per_dimension) +
                                params.host_constants.lbb;
    params.host_constants.interval =
        params.host_constants.tfr - params.host_constants.lbb;
    params.particles_per_dimension = particle_box.particles_per_dimension;
}

void computeBubbleDataSize(Params &params) {
    // Calculate the length of 'rows'.
    // Make it divisible by 32, as that's the warp size.
    params.bubbles.stride =
        params.bubbles.count +
        !!(params.bubbles.count % 32) * (32 - params.bubbles.count % 32);

    // It seems to hold that in 3 dimensions the total number of
    // bubble pairs is 11x and in two dimensions 4x number of bubbles.
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avg_num_neighbors =
        (params.host_constants.dimensionality == 3) ? 12 : 4;
    params.pairs.stride = avg_num_neighbors * params.bubbles.stride;
}

void printStartingParameters(Params &params) {
    printf("---------------Starting parameters---------------\n");
    params.host_constants.print();
    params.host_data.print();
    params.bubbles.print();
    params.pairs.print();
    printf("-------------------------------------------------\n");
}

void allocateCopyConstansToGPU(Params &params) {
    // Allocate and copy constants to GPU
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.device_constants),
                           sizeof(Constants)));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.device_constants),
                         static_cast<void *>(&params.host_constants),
                         sizeof(Constants), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpyToSymbol(d_constants,
                                 static_cast<void *>(&params.device_constants),
                                 sizeof(Constants *)));

    CUDA_CALL(
        cudaEventCreate(&params.snapshot_params.event, cudaEventDisableTiming));

    printRelevantInfoOfCurrentDevice();

    int32_t zero = 0;
    CUDA_CALL(cudaMemcpyToSymbol(d_num_pairs, static_cast<void *>(&zero),
                                 sizeof(int32_t)));
    KERNEL_LAUNCH(initGlobals, params, 0, 0);
}

void reserveDeviceMemore(Params &params) {
    printf("Reserving device memory\n");
    CUDA_CALL(cudaMallocHost(&params.pinned_memory, sizeof(int32_t)));

    // Total memory: memory for bubble data, memory for pair data and memory for
    // temporary arrays
    size_t bytes = params.bubbles.getMemReq();
    bytes += params.pairs.getMemReq();
    bytes += params.getTempMemReq();
    CUDA_ASSERT(cudaMalloc(&params.memory, bytes));

    params.maximums.resize(3 * GRID_SIZE);

    // If we're going to be saving snapshots, allocate enough memory to hold all
    // the device data.
    if (0.0 < params.host_data.snapshot_frequency) {
        params.host_memory.resize(bytes);
    }

    // Each named pointer is setup by these functions to point to
    // a different stride inside the continuous memory blob
    void *pair_start = params.bubbles.setupPointers(params.memory);
    pair_start = params.pairs.setupPointers(pair_start);
    params.setTempPointers(pair_start);

    params.previous_x.resize(params.bubbles.stride);
    params.previous_y.resize(params.bubbles.stride);
    params.previous_z.resize(params.bubbles.stride);
    params.snapshot_params.x0.resize(params.bubbles.stride);
    params.snapshot_params.y0.resize(params.bubbles.stride);
    params.snapshot_params.z0.resize(params.bubbles.stride);

    const uint64_t megs = bytes / (1024 * 1024);
    const uint64_t kilos = (bytes - megs * 1024 * 1024) / 1024;
    bytes = (bytes - megs * 1024 * 1024 - kilos * 1024);
    printf("Allocated %ld MB %ld KB %ld B of global device memory.\n", megs,
           kilos, bytes);
}

void generateStartingData(Params &params, const nlohmann::json &input_json) {
    printf("Generating starting data\n");
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(
        generator, input_json["rngSeed"]["value"]));
    if (params.host_constants.dimensionality == 3) {
        CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.z,
                                                params.bubbles.count));
    }
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.x,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.y,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.rp,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateNormalDouble(
        generator, params.bubbles.r, params.bubbles.count,
        params.host_data.avg_rad, input_json["bubbles"]["radius"]["std"]));
    CURAND_CALL(curandDestroyGenerator(generator));

    KERNEL_LAUNCH(assignDataToBubbles, params, 0, 0,
                  params.particles_per_dimension, params.host_data.avg_rad,
                  params.bubbles);

    // Get the average input surface area and maximum bubble radius
    void *cub_ptr = static_cast<void *>(params.temp_pair2);
    void *cub_output = nullptr;
    void *out =
        static_cast<void *>(&params.host_constants.average_surface_area_in);
    CUDA_CALL(cudaGetSymbolAddress(&cub_output, d_max_radius));
    CUB_LAUNCH(&cub::DeviceReduce::Sum, cub_ptr, params.pairs.getMemReq() / 2,
               params.bubbles.rp, static_cast<double *>(cub_output),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(out, d_max_radius, sizeof(double)));

    out = static_cast<void *>(&params.host_data.max_bubble_radius);
    CUB_LAUNCH(&cub::DeviceReduce::Max, cub_ptr, params.pairs.getMemReq() / 2,
               params.bubbles.r, static_cast<double *>(cub_output),
               params.bubbles.count, (cudaStream_t)0, false);
    CUDA_CALL(cudaMemcpyFromSymbol(out, d_max_radius, sizeof(double)));
}

void performFirstNeigborSearch(Params &params) {
    printf("First neighbor search\n");
    searchNeighbors(params);

    // After searchNeighbors x, y, z, r are correct,
    // but all predicted are trash. pairwiseInteraction always uses
    // predicted values, so copy currents to predicteds
    size_t bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.xp),
                              static_cast<void *>(params.bubbles.x), bytes,
                              cudaMemcpyDefault, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.yp),
                              static_cast<void *>(params.bubbles.y), bytes,
                              cudaMemcpyDefault, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.zp),
                              static_cast<void *>(params.bubbles.z), bytes,
                              cudaMemcpyDefault, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDefault, 0));
}

void performInitialVelocityComputation(Params &params) {
    printf("Calculating initial velocities for Adams-Bashforth-Moulton\n");
    KERNEL_LAUNCH(
        resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
        params.bubbles.dxdto, params.bubbles.dydto, params.bubbles.dzdto,
        params.bubbles.drdto, params.bubbles.dxdtp, params.bubbles.dydtp,
        params.bubbles.dzdtp, params.bubbles.drdtp, params.bubbles.path);
    const uint32_t dyn_shared_mem_bytes =
        params.host_constants.dimensionality * BLOCK_SIZE * sizeof(double);
    KERNEL_LAUNCH(pairwiseInteraction, params, dyn_shared_mem_bytes, 0,
                  params.bubbles, params.pairs, params.temp_d1, false, false);
    KERNEL_LAUNCH(euler, params, 0, 0, params.host_data.time_step,
                  params.bubbles);

    // pairwiseInteraction calculates to predicteds by accumulating values
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

    KERNEL_LAUNCH(pairwiseInteraction, params, dyn_shared_mem_bytes, 0,
                  params.bubbles, params.pairs, params.temp_d1, false, false);

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

void stabilizeAfterCreate(Params &params, const nlohmann::json &input_json) {
    printf("Stabilizing a few rounds after creation\n");
    const int32_t stabilization_steps = input_json["stabilization"]["steps"];
    for (uint32_t i = 0; i < 5; ++i) {
        stabilize(params, stabilization_steps);
    }
}

void scaleAfterStabilize(Params &params, const nlohmann::json &input_json) {
    printf("Scaling the simulation box\n");
    const double bubble_volume = totalVolume(params);
    const double phi = input_json["constants"]["phi"]["value"];
    printf("Current phi: %.9g, target phi: %.9g\n",
           bubble_volume / boxVolume(params), phi);

    KERNEL_LAUNCH(transformPositions, params, 0, 0, true, params.bubbles);

    dvec rel_dim = input_json["box"]["relativeDimensions"];
    double t = bubble_volume / (phi * rel_dim.x * rel_dim.y);
    if (params.host_constants.dimensionality == 3) {
        t /= rel_dim.z;
        t = std::cbrt(t);
    } else {
        t = std::sqrt(t);
        rel_dim.z = 0.0;
    }

    params.host_constants.tfr = dvec(t, t, t) * rel_dim;
    params.host_constants.interval =
        params.host_constants.tfr - params.host_constants.lbb;
    params.host_constants.flow_tfr =
        params.host_constants.interval * params.host_constants.flow_tfr +
        params.host_constants.lbb;
    params.host_constants.flow_lbb =
        params.host_constants.interval * params.host_constants.flow_lbb +
        params.host_constants.lbb;
    params.snapshot_params.interval = params.host_constants.interval;

    double mult = phi * boxVolume(params) / CUBBLE_PI;
    if (params.host_constants.dimensionality == 3) {
        mult = std::cbrt(0.75 * mult);
    } else {
        mult = std::sqrt(mult);
    }
    params.host_constants.bubble_volume_multiplier = mult;

    // Copy the updated constants to GPU
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.device_constants),
                         static_cast<void *>(&params.host_constants),
                         sizeof(Constants), cudaMemcpyDefault));

    KERNEL_LAUNCH(transformPositions, params, 0, 0, false, params.bubbles);

    printf("Current phi: %.9g, target phi: %.9g\n",
           bubble_volume / boxVolume(params), phi);
}

void performSecondNeighborSearch(Params &params,
                                 const nlohmann::json &input_json) {
    printf("Neighbor search after scaling\n");
    searchNeighbors(params);
    // After searchNeighbors r is correct,
    // but rp is trash. pairwiseInteraction always uses
    // predicted values, so copy r to rp
    size_t bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDefault, 0));

    printf("Stabilizing a few rounds after scaling\n");
    const int32_t stabilization_steps = input_json["stabilization"]["steps"];
    for (uint32_t i = 0; i < 5; ++i) {
        stabilize(params, stabilization_steps);
    }
}

void performFinalStabilization(Params &params,
                               const nlohmann::json &input_json) {
    printf("\n=============\nStabilization\n=============\n");
    params.host_data.num_neighbors_searched = 0;
    int32_t num_steps = 0;
    const int32_t failsafe = 500;

    printf("%-7s %-11s %-11s %-11s %-9s\n", "#steps", "dE", "e1", "e2",
           "#searches");
    const double &e1 = params.host_data.energy1;
    const double &e2 = params.host_data.energy2;
    const int32_t stabilization_steps = input_json["stabilization"]["steps"];

    while (true) {
        params.host_data.time_integer = 0;
        params.host_data.time_fraction = 0.0;

        stabilize(params, stabilization_steps);
        double time = ((double)params.host_data.time_integer +
                       params.host_data.time_fraction) *
                      params.host_data.time_scaling_factor;

        double de = std::abs(e2 - e1);
        if (de > 0.0) {
            de *= 2.0 / ((e2 + e1) * time);
        }

        const bool stop = de < input_json["stabilization"]["maxDeltaEnergy"] ||
                          (e2 < 1.0 && de < 0.1);
        if (stop) {
            printf("Final energies:");
            printf("\nbefore: %9.5e", e1);
            printf("\nafter: %9.5e", e2);
            printf("\ndelta: %9.5e", de);
            printf("\ntime: %9.5g\n", time);
            break;
        } else if (num_steps > failsafe) {
            printf("Over %d steps taken and required delta energy not reached. "
                   "Constraints might be too strict.\n",
                   num_steps);
            break;
        } else {
            printf("%-7d ", (num_steps + 1) * stabilization_steps);
            printf("%-9.5e ", de);
            printf("%-9.5e ", e1);
            printf("%-9.5e ", e2);
            printf("%-9ld\n", params.host_data.num_neighbors_searched);
            params.host_data.num_neighbors_searched = 0;
        }

        ++num_steps;
    }

    if (0.0 < params.host_data.snapshot_frequency) {
        // Set starting positions.
        // Avoiding batched copy, because the pointers might not be in order
        int32_t *index = reinterpret_cast<int32_t *>(params.host_memory.data());
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previous_x.data()),
                             static_cast<void *>(params.bubbles.x),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previous_y.data()),
                             static_cast<void *>(params.bubbles.y),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previous_z.data()),
                             static_cast<void *>(params.bubbles.z),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(index),
                             static_cast<void *>(params.bubbles.index),
                             sizeof(int32_t) * params.bubbles.count,
                             cudaMemcpyDefault));

        for (uint64_t i = 0; i < params.bubbles.count; i++) {
            const int32_t ind = index[i];
            params.snapshot_params.x0[ind] = params.previous_x[i];
            params.snapshot_params.y0[ind] = params.previous_y[i];
            params.snapshot_params.z0[ind] = params.previous_z[i];
        }

        // 'Previous' vectors are updated with the previous positions after
        // every snapshot, but since there is nothing previous to the first
        // snapshot, initialize them with the starting positions.
        std::copy(params.snapshot_params.x0.begin(),
                  params.snapshot_params.x0.end(), params.previous_x.begin());
        std::copy(params.snapshot_params.y0.begin(),
                  params.snapshot_params.y0.end(), params.previous_y.begin());
        std::copy(params.snapshot_params.z0.begin(),
                  params.snapshot_params.z0.end(), params.previous_z.begin());
    }

    // Reset wrap counts to 0
    // Avoiding batched memset, because the pointers might not be in order
    size_t bytes = sizeof(int32_t) * params.bubbles.stride;
    CUDA_CALL(cudaMemset(params.bubbles.wrap_count_x, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrap_count_y, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrap_count_z, 0, bytes));

    // Reset errors since integration starts after this
    KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count, false,
                  params.bubbles.error);

    params.host_data.energy1 = totalEnergy(params);
    params.host_data.time_integer = 0;
    params.host_data.time_fraction = 0.0;
    params.host_data.times_printed = 1;
    params.host_data.num_integration_steps = 0;
}

void init(const char *inputFileName, Params &params) {
    printf("==============\nInitialization\n==============\n");
    const auto input_json = parse(inputFileName);

    initializeHostConstants(params, input_json);
    initializeHostData(params, input_json);
    computeSizeOfBox(params, input_json);
    computeBubbleDataSize(params);
    printStartingParameters(params);
    allocateCopyConstansToGPU(params);
    reserveDeviceMemore(params);
    generateStartingData(params, input_json);
    performFirstNeigborSearch(params);
    performInitialVelocityComputation(params);
    stabilizeAfterCreate(params, input_json);
    scaleAfterStabilize(params, input_json);
    performSecondNeighborSearch(params, input_json);
    performFinalStabilization(params, input_json);
}

void simulate(std::string &&inputFileName) {
    Params params;
    init(inputFileName.c_str(), params);

    if (params.host_data.snapshot_frequency > 0.0) {
        saveSnapshot(params);
    }

    bool continue_simulation = true;
    double min_timestep = 9999999.9;
    double max_timestep = -1.0;
    double avg_timestep = 0.0;
    bool reset_errors = false;
    double &ts = params.host_data.time_step;

    printProgress(params, min_timestep, max_timestep, avg_timestep,
                  reset_errors, true);

    const double min_interval =
        3 == params.host_constants.dimensionality
            ? 0.5 * minComponent(params.host_constants.interval)
            : 0.5 * (params.host_constants.interval.x <
                             params.host_constants.interval.y
                         ? params.host_constants.interval.x
                         : params.host_constants.interval.y);

    CUBBLE_PROFILE(true);

    IntegrationParams ip;
    ip.use_gas_exchange = true;
    ip.use_flow = params.host_data.add_flow;
    ip.increment_path = true;
    ip.error_too_large = true;
    ip.max_radius = 0.0;
    ip.max_expansion = 0.0;
    ip.max_error = 0.0;
    ip.h_num_to_be_deleted = static_cast<int32_t *>(params.pinned_memory);

    while (continue_simulation) {
        CUBBLE_PROFILE(false);
        integrate(params, ip);
        continue_simulation = checkEndConditions(params, min_interval);
        trackTimeStep(ts, min_timestep, max_timestep, avg_timestep);
        printProgress(params, min_timestep, max_timestep, avg_timestep,
                      reset_errors, false);
        checkSnapshot(params);

        if (reset_errors) {
            KERNEL_LAUNCH(resetArrays, params, 0, 0, 0.0, params.bubbles.count,
                          false, params.bubbles.error);
            reset_errors = false;
        }

        ++params.host_data.num_steps_in_time_step;
    }

    end(params, min_interval);
    printf("Done\n");
}
} // namespace

namespace cubble {
int32_t run(int32_t argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s inputFile\n\twhere inputFile is the name of the "
               "(.json) file containing the simulation input.\n",
               argv[0]);

        return EXIT_FAILURE;
    }

    int32_t num_gpu = 0;
    CUDA_CALL(cudaGetDeviceCount(&num_gpu));
    if (num_gpu < 1) {
        printf("No CUDA capable devices found.\n");
        return EXIT_FAILURE;
    }

    try {
        simulate(std::string(argv[1]));
    } catch (const std::exception &e) {
        cubble::handleException(std::current_exception());
        return EXIT_FAILURE;
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
} // namespace cubble
