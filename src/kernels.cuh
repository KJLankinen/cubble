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

#include "data_definitions.h"
#include "free_functions.cuh"
#include "vec.h"

#include <stdint.h>

namespace cubble {
__global__ void preIntegrate(double ts, bool useGasExchange, Bubbles bubbles,
                             double *temp1);
__global__ void pairwiseInteraction(Bubbles bubbles, Pairs pairs,
                                    double *overlap, bool useGasExchange,
                                    bool useFlow);
__global__ void postIntegrate(double ts, bool useGasExchange,
                              bool incrementPath, bool useFlow, Bubbles bubbles,
                              double *blockMax, double *overlap,
                              int32_t *toBeDeleted);
__global__ void cellByPosition(int32_t *cellIndices, int32_t *cellSizes,
                               ivec cellDim, Bubbles bubbles);
__global__ void indexByCell(int32_t *cellIndices, int32_t *cellOffsets,
                            int32_t *bubbleIndices, int32_t count);
__global__ void neighborSearch(int32_t numCells, int32_t numNeighborCells,
                               ivec cellDim, int32_t *offsets, int32_t *sizes,
                               int32_t *histogram, int32_t *pairI,
                               int32_t *pairJ, Bubbles bubbles);
__global__ void sortPairs(Bubbles bubbles, Pairs pairs, int32_t *pairI,
                          int32_t *pairJ);
__global__ void countNumNeighbors(Bubbles bubbles, Pairs pairs);
__global__ void reorganizeByIndex(Bubbles bubbles, const int32_t *newIndex);
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int32_t *toBeDeleted);
__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int32_t *toBeDeleted);
__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs, double *energy);
__global__ void euler(double ts, Bubbles bubbles);
__global__ void transformPositions(bool normalize, Bubbles bubbles);
__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles);
__global__ void calculateVolumes(Bubbles bubbles, double *volumes);
__global__ void assignDataToBubbles(uvec bubblesPerDim, double avgRad,
                                    Bubbles bubbles);
__global__ void initGlobals();
template <typename T, typename... Args>
__global__ void resetArrays(T value, int32_t numValues, bool resetGlobals,
                            Args... args) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        setArrayToValue(value, i, args...);
    }

    if (resetGlobals && threadIdx.x + blockIdx.x == 0) {
        resetDeviceGlobals();
    }
}
} // namespace cubble
