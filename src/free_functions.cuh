/*
    Cubble
    Copyright (C) 2024  Juhana Lankinen

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

#include "data_definitions.h"
#include "vec.h"

namespace cubble {
__device__ double correct(int32_t i, double ts, double *pp, double *p,
                          double *v, double *vp, double *old, double *maxErr);
__device__ void addFlowVelocity(Bubbles &bubbles, int32_t i);
__device__ void addWallVelocity(Bubbles &bubbles, int32_t i);
__device__ void comparePair(int32_t idx1, int32_t idx2, int32_t *histogram,
                            int32_t *pairI, int32_t *pairJ, Bubbles &bubbles);
__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);
__device__ dvec wrappedDifference(double x1, double y1, double z1, double x2,
                                  double y2, double z2);
__device__ int32_t getNeighborCellIndex(int32_t cellIdx, ivec dim,
                                        int32_t neighborNum);
__device__ int32_t getCellIdxFromPos(double x, double y, double z,
                                     ivec cellDim);
__device__ int32_t get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ ivec get3DIdxFrom1DIdx(int32_t idx, ivec cellDim);

template <typename T> __device__ T sum(T a, T b) { return a + b; }
template <typename T> __device__ T max(T a, T b) { return a > b ? a : b; }
template <typename T>
__device__ void reduce(T *addr, int32_t warp, T (*f)(T, T)) {
    // Assumes that addr.length() == BLOCK_SIZE
    const int32_t tid = threadIdx.x;
    const int32_t wid = (tid & 31);
#pragma unroll
    for (int32_t i = 0; i < BLOCK_SIZE / 32; i++) {
        if (i == warp) {
            continue;
        }
        addr[tid] = f(addr[tid], addr[wid + i * 32]);
    }
    __syncwarp();

    T temp;
    temp = addr[tid ^ 16];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 8];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 4];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    temp = addr[tid ^ 2];
    __syncwarp();
    addr[tid] = f(addr[tid], temp);
    __syncwarp();

    if (0 == wid) {
        addr[tid] = f(addr[tid], addr[tid + 1]);
    }
}

template <typename T>
__device__ void recursiveReduce(T (*f)(T, T), int32_t idx, T *baseAddr, T *,
                                T *to, int32_t offset, bool flag) {
    if (flag) {
        *to = f(*to, baseAddr[idx + offset * BLOCK_SIZE]);
    }
}

template <typename... Args, typename T>
__device__ void recursiveReduce(T (*f)(T, T), int32_t idx, T *baseAddr, T *temp,
                                T *to, int32_t offset, bool flag,
                                Args... args) {
    recursiveReduce(f, idx, baseAddr, temp, to, offset, flag);
    recursiveReduce(f, idx, baseAddr, args...);
}

template <typename T>
__device__ void recursiveAtomicAdd(T *to, T *val, int32_t, bool flag) {
    if (flag) {
        atomicAdd(to, *val);
    }
}

template <typename... Args, typename T>
__device__ void recursiveAtomicAdd(T *to, T *val, int32_t, bool flag,
                                   Args... args) {
    recursiveAtomicAdd(to, val, 0, flag);
    recursiveAtomicAdd(args...);
}

template <typename... Args, typename T>
__device__ void warpReduceAtomicAddMatching(uint32_t active, int32_t matchOn,
                                            T (*f)(T, T), T *baseAddr,
                                            Args... args) {
    const uint32_t matches = __match_any_sync(active, matchOn);
    const uint32_t lanemask_lt = (1 << (threadIdx.x & 31)) - 1;
    const uint32_t rank = __popc(matches & lanemask_lt);
    const int32_t flt = 32 * (threadIdx.x >> 5);

    if (0 == rank) {
#pragma unroll
        for (int32_t j = 0; j < 32; j++) {
            if (!!(matches & 1 << j)) {
                recursiveReduce(f, j + flt, baseAddr, args...);
            }
        }
        recursiveAtomicAdd(args...);
    }
}

__device__ void resetDeviceGlobals();
template <typename T>
__device__ void swapValues(int32_t from, int32_t to, T *arr) {
    arr[to] = arr[from];
}
template <typename T, typename... Args>
__device__ void swapValues(int32_t from, int32_t to, T *arr, Args... args) {
    swapValues(from, to, arr);
    swapValues(from, to, args...);
}

template <typename T>
__device__ void setArrayToValue(T value, int32_t idx, T *array) {
    array[idx] = value;
}
template <typename T, typename... Args>
__device__ void setArrayToValue(T value, int32_t idx, T *array, Args... args) {
    setArrayToValue(value, idx, array);
    setArrayToValue(value, idx, args...);
}
} // namespace cubble
