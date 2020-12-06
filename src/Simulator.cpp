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

#include "DataDefinitions.h"
#include "KernelWrapper.h"
#include "Util.h"
#include "Vec.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <fstream>
#include <functional>
#include <mpi.h>
#include <nvToolsExt.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

namespace {
using namespace cubble;
void externalNeighborSearch(Params &params) {
    std::array<int, 26> oppositeAreas;
    if (3 == params.hostConstants.dimensionality) {
        oppositeAreas[0] = 1;
        oppositeAreas[1] = 0;
        oppositeAreas[2] = 3;
        oppositeAreas[3] = 2;
        oppositeAreas[4] = 5;
        oppositeAreas[5] = 4;
        oppositeAreas[6] = 8;
        oppositeAreas[7] = 9;
        oppositeAreas[8] = 6;
        oppositeAreas[9] = 7;
        oppositeAreas[10] = 12;
        oppositeAreas[11] = 13;
        oppositeAreas[12] = 10;
        oppositeAreas[13] = 11;
        oppositeAreas[14] = 16;
        oppositeAreas[15] = 17;
        oppositeAreas[16] = 14;
        oppositeAreas[17] = 15;
        oppositeAreas[18] = 25;
        oppositeAreas[19] = 22;
        oppositeAreas[20] = 23;
        oppositeAreas[21] = 24;
        oppositeAreas[22] = 19;
        oppositeAreas[23] = 20;
        oppositeAreas[24] = 21;
        oppositeAreas[25] = 18;
    } else {
        oppositeAreas[0] = 1;
        oppositeAreas[1] = 0;
        oppositeAreas[2] = 3;
        oppositeAreas[3] = 2;
        oppositeAreas[4] = 6;
        oppositeAreas[5] = 7;
        oppositeAreas[6] = 4;
        oppositeAreas[7] = 5;
    }

    ivec cellDim = params.neighborSearchData.cellDim;
    const int numCells = params.neighborSearchData.numCells;
    int nAreas = 8;
    int nSurfaceCells = 2 * (cellDim.x + cellDim.y) + 4;
    if (3 == params.hostConstants.dimensionality) {
        nSurfaceCells = 2 * (cellDim.x * cellDim.y + cellDim.x * cellDim.z +
                             cellDim.y * cellDim.z) +
                        4 * (cellDim.x + cellDim.y + cellDim.z) + 8;
        nAreas = 26;
    }
    const int numCellsToSearch = nAreas + 1;

    // Per surface cell data
    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.tempPair1), 0,
                              4 * nSurfaceCells * sizeof(int), 0));
    int *surfaceCells = params.tempPair1;
    int *surfaceCellSizes = surfaceCells + nSurfaceCells;
    int *surfaceCellOffsets = surfaceCellSizes + nSurfaceCells;
    int *bubbleCountPerArea = surfaceCellOffsets + nSurfaceCells;
    // HACK
    // This is very unsafe and frankly disgusting. These values were calculated
    // to these addresses by the neighborSearch function, but if these are
    // changed there, everything breaks here.
    int *cellOffsets = params.neighborSearchData.cellOffsets;
    int *cellSizes = params.neighborSearchData.cellSizes;

    launchFindSurfaceCells(params, numCells, surfaceCells, cellSizes,
                           surfaceCellSizes, bubbleCountPerArea, cellDim);

    void *cubPtr = static_cast<void *>(params.tempPair2);
    const uint64_t maxCubMem = params.pairs.getMemReq() / 2;
    cubExclusiveSum(cubPtr, maxCubMem, surfaceCellSizes, surfaceCellOffsets,
                    nSurfaceCells, 0, false);

    std::array<int, 27> bubbleCounts;
    CUDA_CALL(cudaMemcpy(static_cast<void *>(bubbleCounts.data()),
                         static_cast<void *>(bubbleCountPerArea),
                         sizeof(int) * 27, cudaMemcpyDefault));

#ifndef NDEBUG
    int temp = 0;
    for (int i = 0; i < nAreas; i++) {
        temp += bubbleCounts[i];
    }
    assert(temp == bubbleCounts[26] && "Bubblecount totals don't match.");
#endif

    std::array<int, 26> cellCounts;
    if (3 == params.hostConstants.dimensionality) {
        cellCounts[0] = cellDim.y * cellDim.z;
        cellCounts[1] = cellDim.y * cellDim.z;
        cellCounts[2] = cellDim.x * cellDim.z;
        cellCounts[3] = cellDim.x * cellDim.z;
        cellCounts[4] = cellDim.x * cellDim.y;
        cellCounts[5] = cellDim.x * cellDim.y;
        cellCounts[6] = cellDim.x;
        cellCounts[7] = cellDim.x;
        cellCounts[8] = cellDim.x;
        cellCounts[9] = cellDim.x;
        cellCounts[10] = cellDim.y;
        cellCounts[11] = cellDim.y;
        cellCounts[12] = cellDim.y;
        cellCounts[13] = cellDim.y;
        cellCounts[14] = cellDim.z;
        cellCounts[15] = cellDim.z;
        cellCounts[16] = cellDim.z;
        cellCounts[17] = cellDim.z;
        cellCounts[18] = 1;
        cellCounts[19] = 1;
        cellCounts[20] = 1;
        cellCounts[21] = 1;
        cellCounts[22] = 1;
        cellCounts[23] = 1;
        cellCounts[24] = 1;
        cellCounts[25] = 1;
    } else {
        cellCounts[0] = cellDim.y;
        cellCounts[1] = cellDim.y;
        cellCounts[2] = cellDim.x;
        cellCounts[3] = cellDim.x;
        cellCounts[4] = 1;
        cellCounts[5] = 1;
        cellCounts[6] = 1;
        cellCounts[7] = 1;
    }

    std::array<int, 27> incomingBubbleCounts;
    incomingBubbleCounts[26] = 0;
    for (int i = 0; i < nAreas; i++) {
        const int oppositeArea = oppositeAreas[i];
        const int dstProc = params.areaToProcessorMap[i];
        const int srcProc = params.areaToProcessorMap[oppositeArea];
        const int tag = 1337;
        MPI_Status status;
        int rc = MPI_Sendrecv(
            static_cast<void *>(&bubbleCounts[i]), 1, MPI_INT, dstProc, tag,
            static_cast<void *>(&incomingBubbleCounts[oppositeArea]), 1,
            MPI_INT, srcProc, tag, params.comm, &status);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }
        incomingBubbleCounts[26] += incomingBubbleCounts[oppositeArea];
    }

    std::vector<double> incomingX(incomingBubbleCounts[26]);
    std::vector<double> incomingY(incomingBubbleCounts[26]);
    std::vector<double> incomingZ(incomingBubbleCounts[26]);
    std::vector<double> incomingR(incomingBubbleCounts[26]);
    std::vector<int> incomingIdx(incomingBubbleCounts[26]);
    std::vector<int> incomingSize(nSurfaceCells);

    uint64_t bytes = (4 * sizeof(double) + sizeof(int)) * bubbleCounts[26] +
                     sizeof(int) * nSurfaceCells;
    void *outgoingMem = nullptr;
    CUDA_ASSERT(cudaMalloc(&outgoingMem, bytes));
    double *x = static_cast<double *>(outgoingMem);
    double *y = x + bubbleCounts[26];
    double *z = y + bubbleCounts[26];
    double *r = z + bubbleCounts[26];
    int *idx = reinterpret_cast<int *>(r + bubbleCounts[26]);
    int *sizes = idx + bubbleCounts[26];

    launchGatherSurfaceBubbles(params, nSurfaceCells, surfaceCells,
                               surfaceCellOffsets, cellSizes, cellOffsets, x, y,
                               z, r, idx, sizes);

    int outBubOffset = 0;
    int outCellOffset = 0;
    int inBubOffset = 0;
    int inCellOffset = 0;
    std::vector<char> outData;
    std::vector<char> inData;
    for (int i = 0; i < nAreas; i++) {
        const int oppositeArea = oppositeAreas[i];
        const int dstProc = params.areaToProcessorMap[i];
        const int srcProc = params.areaToProcessorMap[oppositeArea];
        const int tag = 1337;
        MPI_Status status;

        const int outBubCount = bubbleCounts[i];
        const int inBubCount = incomingBubbleCounts[oppositeArea];
        const int bytesPerBubble = 4 * sizeof(double) + sizeof(int);
        const int outBytes =
            bytesPerBubble * outBubCount + cellCounts[i] * sizeof(int);
        const int inBytes = bytesPerBubble * inBubCount +
                            cellCounts[oppositeArea] * sizeof(int);

        if (outData.size() < (uint32_t)outBytes) {
            outData.resize(outBytes);
        }

        if (inData.size() < (uint32_t)inBytes) {
            inData.resize(inBytes);
        }

        char *dst = outData.data();
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(x + outBubOffset),
                             sizeof(double) * outBubCount, cudaMemcpyDefault));
        dst += sizeof(double) * outBubCount;
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(y + outBubOffset),
                             sizeof(double) * outBubCount, cudaMemcpyDefault));
        dst += sizeof(double) * outBubCount;
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(z + outBubOffset),
                             sizeof(double) * outBubCount, cudaMemcpyDefault));
        dst += sizeof(double) * outBubCount;
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(r + outBubOffset),
                             sizeof(double) * outBubCount, cudaMemcpyDefault));
        dst += sizeof(double) * outBubCount;
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(idx + outBubOffset),
                             sizeof(int) * outBubCount, cudaMemcpyDefault));
        dst += sizeof(int) * outBubCount;
        CUDA_CALL(cudaMemcpy(static_cast<void *>(dst),
                             static_cast<void *>(sizes + outCellOffset),
                             sizeof(int) * cellCounts[i], cudaMemcpyDefault));

        outBubOffset += outBubCount;
        outCellOffset += cellCounts[i];

        int rc = MPI_Sendrecv(static_cast<void *>(outData.data()), outBytes,
                              MPI_CHAR, dstProc, tag,
                              static_cast<void *>(inData.data()), inBytes,
                              MPI_CHAR, srcProc, tag, params.comm, &status);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        char *src = inData.data();
        memcpy(static_cast<void *>(&incomingX[inBubOffset]),
               static_cast<void *>(src), sizeof(double) * inBubCount);
        src += sizeof(double) * inBubCount;
        memcpy(static_cast<void *>(&incomingY[inBubOffset]),
               static_cast<void *>(src), sizeof(double) * inBubCount);
        src += sizeof(double) * inBubCount;
        memcpy(static_cast<void *>(&incomingZ[inBubOffset]),
               static_cast<void *>(src), sizeof(double) * inBubCount);
        src += sizeof(double) * inBubCount;
        memcpy(static_cast<void *>(&incomingR[inBubOffset]),
               static_cast<void *>(src), sizeof(double) * inBubCount);
        src += sizeof(double) * inBubCount;
        memcpy(static_cast<void *>(&incomingIdx[inBubOffset]),
               static_cast<void *>(src), sizeof(int) * inBubCount);
        src += sizeof(int) * inBubCount;
        memcpy(static_cast<void *>(&incomingSize[inCellOffset]),
               static_cast<void *>(src),
               sizeof(int) * cellCounts[oppositeArea]);

        inBubOffset += inBubCount;
        inCellOffset += cellCounts[oppositeArea];
    }

    bytes = (4 * sizeof(double) + sizeof(int)) * incomingBubbleCounts[26] +
            2 * sizeof(int) * nSurfaceCells;
    void *surfaceMem = nullptr;
    CUDA_ASSERT(cudaMalloc(&surfaceMem, bytes));

    SurfaceData surfaceData;
    surfaceData.x = static_cast<double *>(surfaceMem);
    surfaceData.y = surfaceData.x + incomingBubbleCounts[26];
    surfaceData.z = surfaceData.y + incomingBubbleCounts[26];
    surfaceData.r = surfaceData.z + incomingBubbleCounts[26];
    surfaceData.idx =
        reinterpret_cast<int *>(surfaceData.r + incomingBubbleCounts[26]);
    surfaceData.cellSizes = surfaceData.idx + incomingBubbleCounts[26];
    surfaceData.cellOffsets = surfaceData.cellSizes + nSurfaceCells;

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.x),
                         static_cast<void *>(incomingX.data()),
                         sizeof(double) * incomingX.size(), cudaMemcpyDefault));

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.y),
                         static_cast<void *>(incomingY.data()),
                         sizeof(double) * incomingY.size(), cudaMemcpyDefault));

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.z),
                         static_cast<void *>(incomingZ.data()),
                         sizeof(double) * incomingZ.size(), cudaMemcpyDefault));

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.r),
                         static_cast<void *>(incomingR.data()),
                         sizeof(double) * incomingR.size(), cudaMemcpyDefault));

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.idx),
                         static_cast<void *>(incomingIdx.data()),
                         sizeof(int) * incomingIdx.size(), cudaMemcpyDefault));

    CUDA_CALL(cudaMemcpy(static_cast<void *>(surfaceData.cellSizes),
                         static_cast<void *>(incomingSize.data()),
                         sizeof(int) * incomingSize.size(), cudaMemcpyDefault));

    cubExclusiveSum(cubPtr, maxCubMem, surfaceData.cellSizes,
                    surfaceData.cellOffsets, nSurfaceCells, 0, false);

    setIncomingExternalPairsToZero();

    // Allocate an estimate amount of temp memory for the bubble neigbors
    int stride = 11 * incomingBubbleCounts[26];
    void *tempMemory = nullptr;
    const uint64_t tempMemSize = stride * sizeof(int) * 5;
    CUDA_ASSERT(cudaMalloc(&tempMemory, tempMemSize));
    CUDA_CALL(cudaMemset(tempMemory, 0, tempMemSize));

    int *tempPair1 = reinterpret_cast<int *>(tempMemory);
    int *tempPair2 = tempPair1 + stride;
    int *procNum = tempPair2 + stride;
    int *processorSizes = procNum + stride;
    int *processorOffsets = processorSizes + params.nProcs;

    launchNeighborSearch(params, nSurfaceCells, false, numCellsToSearch,
                         cellDim, cellOffsets, cellSizes, processorSizes,
                         tempPair1, tempPair2, surfaceData, surfaceCells,
                         procNum);

    stride = 0;
    getIncomingExternalPairs(static_cast<void *>(&stride), sizeof(int));
    ExternalBubbles &ib = params.incomingBubbles;
    const uint64_t totalBytes = (7 * sizeof(double) + 2 * sizeof(int)) * stride;
    if (ib.totalBytes < totalBytes || ib.memory == nullptr) {
        if (ib.memory != nullptr) {
            CUDA_CALL(cudaFree(ib.memory));
        }
        CUDA_ASSERT(cudaMalloc(&ib.memory, totalBytes));
        ib.totalBytes = totalBytes;

        ib.data.x = static_cast<double *>(ib.memory);
        ib.data.y = ib.data.x + stride;
        ib.data.z = ib.data.y + stride;
        ib.data.r = ib.data.z + stride;
        ib.data.vx = ib.data.r + stride;
        ib.data.vy = ib.data.vx + stride;
        ib.data.vz = ib.data.vy + stride;
        ib.data.internalIdx = reinterpret_cast<int *>(ib.data.vz + stride);
        ib.data.externalIdx = ib.data.internalIdx + stride;
    }

    cubInclusiveSum(cubPtr, maxCubMem, processorSizes, processorOffsets,
                    params.nProcs, 0, false);

    launchSortExternalPairs(params, tempPair1, tempPair2, processorOffsets,
                            procNum, ib.data);

    const uint32_t nProcs = params.nProcs;
    if (ib.pairsPerProc.size() < nProcs || ib.offsetsPerProc.size() < nProcs) {
        ib.pairsPerProc.resize(params.nProcs);
        ib.offsetsPerProc.resize(params.nProcs);
    }
    CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.offsetsPerProc.data()),
                         static_cast<void *>(processorOffsets),
                         sizeof(int) * ib.offsetsPerProc.size(),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.pairsPerProc.data()),
                         static_cast<void *>(processorSizes),
                         sizeof(int) * ib.pairsPerProc.size(),
                         cudaMemcpyDefault));

    std::vector<char> rcvBuffer(stride * sizeof(int));
    std::vector<char> sendBuffer(ib.pairsPerProc[0] * sizeof(int));
    int totalPairs = 0;
    uint64_t totalIncomingBytes = 0;
    ExternalBubbles &ob = params.outgoingBubbles;
    ob.pairsPerProc.resize(params.nProcs);
    ob.offsetsPerProc.resize(params.nProcs);
    for (int i = 0; i < params.nProcs; i++) {
        uint64_t bytesToSend = ib.pairsPerProc[i] * sizeof(int);
        if (sendBuffer.size() < bytesToSend) {
            sendBuffer.resize(2 * bytesToSend);
        }

        // Get num incoming bytes
        uint64_t incomingBytes = 0;
        const int dstProc = i;
        const int srcProc = i;
        const int tag = 1337;
        MPI_Status status;
        int rc = MPI_Sendrecv(
            static_cast<void *>(&bytesToSend), 1, MPI_UNSIGNED_LONG, dstProc,
            tag, static_cast<void *>(&incomingBytes), 1, MPI_UNSIGNED_LONG,
            srcProc, tag, params.comm, &status);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        if (rcvBuffer.size() < incomingBytes + totalIncomingBytes) {
            rcvBuffer.resize(2 * rcvBuffer.size());
        }

        CUDA_CALL(cudaMemcpy(
            static_cast<void *>(sendBuffer.data()),
            static_cast<void *>(ib.data.externalIdx + ib.offsetsPerProc[i]),
            bytesToSend, cudaMemcpyDefault));

        rc = MPI_Sendrecv(
            static_cast<void *>(sendBuffer.data()), bytesToSend, MPI_CHAR,
            dstProc, tag,
            static_cast<void *>(rcvBuffer.data() + totalIncomingBytes),
            incomingBytes, MPI_CHAR, srcProc, tag, params.comm, &status);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        totalIncomingBytes += incomingBytes;
        const int numPairs = static_cast<int>(incomingBytes / sizeof(int));
        ob.pairsPerProc[i] = numPairs;
        ob.offsetsPerProc[i] = totalPairs;
        totalPairs += numPairs;
    }

    // Allocate memory to hold the outgoingBubbles.data
    const uint64_t outgoingBytes =
        (7 * sizeof(double) + 2 * sizeof(int)) * totalPairs;
    if (ob.totalBytes < outgoingBytes || ob.memory == nullptr) {
        if (ob.memory != nullptr) {
            CUDA_CALL(cudaFree(ob.memory));
        }

        CUDA_ASSERT(cudaMalloc(&ob.memory, outgoingBytes));
        ob.totalBytes = outgoingBytes;

        ob.data.x = static_cast<double *>(ob.memory);
        ob.data.y = ob.data.x + totalPairs;
        ob.data.z = ob.data.y + totalPairs;
        ob.data.r = ob.data.z + totalPairs;
        ob.data.vx = ob.data.r + totalPairs;
        ob.data.vy = ob.data.vx + totalPairs;
        ob.data.vz = ob.data.vy + totalPairs;
        ob.data.internalIdx = reinterpret_cast<int *>(ob.data.vz + totalPairs);
        ob.data.externalIdx = ob.data.internalIdx + totalPairs;
    }

    // Copy the received data to internalIdx
    CUDA_CALL(cudaMemcpy(static_cast<void *>(ob.data.internalIdx),
                         static_cast<void *>(rcvBuffer.data()),
                         sizeof(int) * totalPairs, cudaMemcpyDefault));

    setOutgoingExternalPairs(static_cast<void *>(&totalPairs), sizeof(int));
    CUDA_CALL(cudaFree(tempMemory));
    CUDA_CALL(cudaFree(outgoingMem));
    CUDA_CALL(cudaFree(surfaceMem));
}

void searchNeighbors(Params &params) {
    nvtxRangePush("Neighbors");
    params.hostData.numNeighborsSearched++;

    void *cubPtr = static_cast<void *>(params.pairs.j);
    uint64_t maxCubMem = params.pairs.getMemReq() / 2;

    setNumToBeDeletedToZero();

    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.pairs.i), 0,
                              params.pairs.getMemReq(), 0));

    int *procSizes = params.pairs.i;
    int *procGlobalOffsets = procSizes + params.nProcs;
    int *procLocalOffsets = procGlobalOffsets + params.nProcs;
    int *movedIndices = procLocalOffsets + params.nProcs;
    int *procNums = movedIndices + params.bubbles.stride;

    launchWrapOverPeriodicBoundaries(params, movedIndices, procNums, procSizes);

    if (params.nProcs > 1) {
        // position and current & old velocity for x, y, z & r
        // = 3 * 4 = 12 doubles
        // path & error = 2 doubles, totaling 14 doubles
        // wrapcount for x, y & z = 3 ints
        // ints and doubles are mixed in the array, so add a padding int,
        // s.t. the byte count is divisible by 8, i.e. sizeof(double)
        const int bytesPerBubble = 14 * sizeof(double) + 4 * sizeof(int);

        int numToMove = 0;
        getNumToBeDeleted(static_cast<void *>(&numToMove), sizeof(int), false);

        void *movedData = nullptr;
        void *receivedData = nullptr;
        // This is just a wild guess
        uint64_t maxReceivedBytes = params.bubbles.count / 10 * bytesPerBubble;
        CUDA_ASSERT(cudaMalloc(&receivedData, maxReceivedBytes));
        std::vector<int> sizes(params.nProcs);
        std::vector<int> offsets(params.nProcs);
        if (numToMove > 0) {
            cubExclusiveSum(cubPtr, maxCubMem, procSizes, procGlobalOffsets,
                            params.nProcs, 0, false);
            CUDA_CALL(cudaMemcpy(static_cast<void *>(procLocalOffsets),
                                 static_cast<void *>(procSizes),
                                 sizeof(int) * params.nProcs,
                                 cudaMemcpyDefault));

            CUDA_ASSERT(cudaMalloc(&movedData, numToMove * bytesPerBubble));

            setNumToBeDeletedToZero();

            launchGatherAndDeleteMovedBubbles(
                params, numToMove, bytesPerBubble, procSizes, procGlobalOffsets,
                procLocalOffsets, movedIndices, procNums,
                static_cast<char *>(movedData));

            params.bubbles.count -= numToMove;

            CUDA_CALL(cudaMemcpy(static_cast<void *>(sizes.data()),
                                 static_cast<void *>(procSizes),
                                 sizeof(int) * sizes.size(),
                                 cudaMemcpyDefault));

            CUDA_CALL(cudaMemcpy(static_cast<void *>(offsets.data()),
                                 static_cast<void *>(procGlobalOffsets),
                                 sizeof(int) * offsets.size(),
                                 cudaMemcpyDefault));
        } else {
            // Make sure these contain zeros
            for (int i = 0; i < params.nProcs; i++) {
                sizes[i] = 0;
                offsets[i] = 0;
            }
        }

        std::vector<int> receivedProcNums;
        int numReceivedBubbles = 0;
        uint64_t totalBytesReceived = 0;
        char *src = static_cast<char *>(movedData);
        char *dst = static_cast<char *>(receivedData);
        for (int i = 0; i < params.nProcs; i++) {
            uint64_t bytesToSend = bytesPerBubble * sizes[i];
            uint64_t bytesReceived = 0;

            const int dstProc = i;
            const int srcProc = i;
            const int tag = 1337;
            MPI_Status status;
            int rc = MPI_Sendrecv(
                static_cast<void *>(&bytesToSend), 1, MPI_UNSIGNED_LONG,
                dstProc, tag, static_cast<void *>(&bytesReceived), 1,
                MPI_UNSIGNED_LONG, srcProc, tag, params.comm, &status);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }

            if (bytesReceived + totalBytesReceived > maxReceivedBytes) {
                // If the data were receiving is more than we have allocated
                // memory for, allocate double the old size
                std::vector<char> tempData(totalBytesReceived);
                CUDA_CALL(cudaMemcpy(static_cast<void *>(tempData.data()),
                                     receivedData, tempData.size(),
                                     cudaMemcpyDefault));
                CUDA_CALL(cudaFree(receivedData));
                maxReceivedBytes *= 2;
                CUDA_ASSERT(cudaMalloc(&receivedData, maxReceivedBytes));
                CUDA_CALL(cudaMemcpy(receivedData,
                                     static_cast<void *>(tempData.data()),
                                     tempData.size(), cudaMemcpyDefault));
                // Reset the destination pointer
                dst = static_cast<char *>(receivedData) + totalBytesReceived;
            }

            rc = MPI_Sendrecv(static_cast<void *>(src), bytesToSend, MPI_CHAR,
                              dstProc, tag, static_cast<void *>(dst),
                              bytesReceived, MPI_CHAR, srcProc, tag,
                              params.comm, &status);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }

            src += bytesToSend;
            dst += bytesReceived;
            totalBytesReceived += bytesReceived;
            sizes[i] = static_cast<int>(bytesReceived / bytesPerBubble);
            if (i > 0) {
                offsets[i] = sizes[i - 1] + offsets[i - 1];
            } else {
                offsets[i] = 0;
            }
            numReceivedBubbles += sizes[i];

            for (int j = 0; j < sizes[i]; j++) {
                receivedProcNums.push_back(i);
            }
        }

        // TODO handle this more gracefully
        assert((int)params.bubbles.stride >=
                   params.bubbles.count + numReceivedBubbles &&
               "Not enough memory to hold all the bubbles");

        if (numReceivedBubbles > 0) {
            CUDA_CALL(cudaMemcpy(static_cast<void *>(procSizes),
                                 static_cast<void *>(sizes.data()),
                                 sizeof(int) * sizes.size(),
                                 cudaMemcpyDefault));

            CUDA_CALL(cudaMemcpy(static_cast<void *>(procLocalOffsets),
                                 static_cast<void *>(sizes.data()),
                                 sizeof(int) * sizes.size(),
                                 cudaMemcpyDefault));

            CUDA_CALL(cudaMemcpy(static_cast<void *>(procGlobalOffsets),
                                 static_cast<void *>(offsets.data()),
                                 sizeof(int) * offsets.size(),
                                 cudaMemcpyDefault));

            CUDA_CALL(cudaMemcpy(static_cast<void *>(procNums),
                                 static_cast<void *>(receivedProcNums.data()),
                                 sizeof(int) * receivedProcNums.size(),
                                 cudaMemcpyDefault));

            setNumToBeDeletedToZero();

            launchDistributeReceivedBubbles(
                params, numReceivedBubbles, bytesPerBubble, procSizes,
                procGlobalOffsets, procLocalOffsets, procNums, dst);

            params.bubbles.count += numReceivedBubbles;
        }

        if (movedData != nullptr) {
            CUDA_CALL(cudaFree(movedData));
        }

        if (receivedData != nullptr) {
            CUDA_CALL(cudaFree(receivedData));
        }
    }

    // Reset memory
    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.pairs.i), 0,
                              params.pairs.getMemReq(), 0));
    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.tempI), 0,
                              sizeof(int) * params.bubbles.stride, 0));
    CUDA_CALL(cudaMemsetAsync(static_cast<void *>(params.tempD1), 0,
                              sizeof(int) * 2 * params.bubbles.stride, 0));

    // Minimum size of cell is twice the sum of the skin and max bubble radius
    ivec cellDim = (params.hostConstants.interval /
                    (2 * (params.hostData.maxBubbleRadius +
                          params.hostConstants.skinRadius)))
                       .getFloor()
                       .asType<int>();
    cellDim.x = std::max(cellDim.x, 1);
    cellDim.y = std::max(cellDim.y, 1);
    cellDim.z = std::max(cellDim.z, 1);
    const int numCells = cellDim.x * cellDim.y * cellDim.z;
    params.neighborSearchData.cellDim = cellDim;
    params.neighborSearchData.numCells = numCells;

    // Per cell data
    int *cellOffsets = params.tempI;
    int *cellSizes = cellOffsets + numCells;
    params.neighborSearchData.cellOffsets = cellOffsets;
    params.neighborSearchData.cellSizes = cellSizes;

    // Per bubble data
    int *cellIndices = params.pairs.i;
    int *bubbleIndices = cellIndices + params.bubbles.stride;
    int *histogram = bubbleIndices + params.bubbles.stride;

    launchCellByPosition(params, cellIndices, cellSizes, cellDim);
    cubInclusiveSum(cubPtr, maxCubMem, cellSizes, cellOffsets, numCells, 0,
                    false);
    launchIndexByCell(params, cellIndices, cellOffsets, bubbleIndices,
                      params.bubbles.count);
    {
        launchReorganizeByIndex(params, const_cast<const int *>(bubbleIndices));
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
        params.bubbles.dxdto = params.bubbles.flowVx;
        params.bubbles.flowVx = swapper;

        int *swapperI = params.bubbles.index;
        params.bubbles.index = params.bubbles.wrapCountZ;
        params.bubbles.wrapCountZ = params.bubbles.wrapCountY;
        params.bubbles.wrapCountY = params.bubbles.wrapCountX;
        params.bubbles.wrapCountX = params.bubbles.numNeighbors;
        params.bubbles.numNeighbors = swapperI;
    }

    setNumPairsToZero();

    int numCellsToSearch = 5;
    if (3 == params.hostConstants.dimensionality) {
        numCellsToSearch = 14;
    }

    SurfaceData surfaceData;
    launchNeighborSearch(params, numCells, true, numCellsToSearch, cellDim,
                         cellOffsets, cellSizes, histogram, params.tempPair1,
                         params.tempPair2, surfaceData, cellOffsets,
                         cellOffsets);

    getNumPairs(static_cast<void *>(&params.pairs.count), sizeof(int));

    cubInclusiveSum(cubPtr, maxCubMem, histogram, params.bubbles.numNeighbors,
                    params.bubbles.count, 0, false);

    launchSortPairs(params);

    CUDA_CALL(cudaMemset(static_cast<void *>(params.bubbles.numNeighbors), 0,
                         params.bubbles.count * sizeof(int)));

    launchCountNumNeighbors(params);

    if (params.nProcs > 1) {
        externalNeighborSearch(params);
    }

    nvtxRangePop();
}

void step(Params &params, IntegrationParams &ip) {
    nvtxRangePush("Integration step");

    launchInitGlobals(params);
    launchPreIntegrate(params, ip);

    if (params.nProcs > 1) {
        // If the data was packed per processor in the ExternalBubbles (i.e. all
        // data of proc N first, then all for proc N + 1), instead of packing
        // all xs for all processors, then all ys and so on, this could be made
        // more efficient, as there'd be only one memcpy before and after the
        // MPI_Sendrecv. That would make the gathering and externalPairwise
        // kernels more complicated, however, so in the interest of development
        // time, this has been implemented like this.
        ExternalBubbles &ib = params.incomingBubbles;
        ExternalBubbles &ob = params.outgoingBubbles;
        launchGatherOutgoingBubbles(params);

        const uint64_t bytesPerPair = 7 * sizeof(double);
        const uint64_t totalIncomingBytes =
            (ib.pairsPerProc.back() + ib.offsetsPerProc.back()) * bytesPerPair;

        if (params.incomingBubbleData.size() < totalIncomingBytes) {
            params.incomingBubbleData.resize(totalIncomingBytes);
        }

        const uint64_t totalOutgoingBytes =
            (ob.pairsPerProc.back() + ob.offsetsPerProc.back()) * bytesPerPair;

        if (params.outgoingBubbleData.size() < totalOutgoingBytes) {
            params.outgoingBubbleData.resize(totalOutgoingBytes);
        }

        for (int i = 0; i < params.nProcs; i++) {
            char *dst = params.incomingBubbleData.data() +
                        bytesPerPair * ib.offsetsPerProc[i];
            const uint64_t incomingBytes = bytesPerPair * ib.pairsPerProc[i];

            char *src = params.outgoingBubbleData.data() +
                        bytesPerPair * ob.offsetsPerProc[i];
            const uint64_t outgoingBytes = bytesPerPair * ob.pairsPerProc[i];

            // If there's no correspondence at all between these processors,
            // skip the whole deal (the other processor does that as well,
            // because it knows the incoming and outgoing amounts as well since
            // it is performing this same very check).
            if (outgoingBytes == 0 && incomingBytes == 0) {
                continue;
            }

            if (outgoingBytes > 0) {
                const uint64_t offset = ob.offsetsPerProc[i];
                const uint64_t bytes = sizeof(double) * ob.pairsPerProc[i];
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src),
                                     static_cast<void *>(ob.data.x + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + bytes),
                                     static_cast<void *>(ob.data.y + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + 2 * bytes),
                                     static_cast<void *>(ob.data.z + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + 3 * bytes),
                                     static_cast<void *>(ob.data.r + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + 4 * bytes),
                                     static_cast<void *>(ob.data.vx + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + 5 * bytes),
                                     static_cast<void *>(ob.data.vy + offset),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(src + 6 * bytes),
                                     static_cast<void *>(ob.data.vz + offset),
                                     bytes, cudaMemcpyDefault));
            }

            const int dstProc = i;
            const int srcProc = i;
            const int tag = 1337;
            MPI_Status status;
            int rc = MPI_Sendrecv(static_cast<void *>(src), outgoingBytes,
                                  MPI_CHAR, dstProc, tag,
                                  static_cast<void *>(dst), incomingBytes,
                                  MPI_CHAR, srcProc, tag, params.comm, &status);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }

            if (incomingBytes > 0) {
                const uint64_t offset = ib.offsetsPerProc[i];
                const uint64_t bytes = sizeof(double) * ib.pairsPerProc[i];
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.x + offset),
                                     static_cast<void *>(dst), bytes,
                                     cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.y + offset),
                                     static_cast<void *>(dst + bytes), bytes,
                                     cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.z + offset),
                                     static_cast<void *>(dst + 2 * bytes),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.r + offset),
                                     static_cast<void *>(dst + 3 * bytes),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.vx + offset),
                                     static_cast<void *>(dst + 4 * bytes),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.vy + offset),
                                     static_cast<void *>(dst + 5 * bytes),
                                     bytes, cudaMemcpyDefault));
                CUDA_CALL(cudaMemcpy(static_cast<void *>(ib.data.vz + offset),
                                     static_cast<void *>(dst + 6 * bytes),
                                     bytes, cudaMemcpyDefault));
            }
        }
    }
    const uint32_t dynSharedMemBytes =
        (params.hostConstants.dimensionality + ip.useGasExchange * 4 +
         ip.useFlow * params.hostConstants.dimensionality) *
        BLOCK_SIZE * sizeof(double);
    launchPairwiseInteraction(params, ip, dynSharedMemBytes);

    if (params.nProcs > 1) {
        launchExternalPairwiseInteraction(params, ip);
        std::array<double, 4> areaTotals;
        getAreaTotals(areaTotals);

        if (params.rank == 0) {
            std::array<double, 4> temp;
            // Receive totals from all other processors
            for (int i = 1; i < params.nProcs; i++) {
                const int tag = 1337;
                MPI_Status status;
                int rc = MPI_Recv(static_cast<void *>(temp.data()), 4,
                                  MPI_DOUBLE, i, tag, params.comm, &status);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
                areaTotals[0] += temp[0];
                areaTotals[1] += temp[1];
                areaTotals[2] += temp[2];
                areaTotals[3] += temp[3];
            }
        } else {
            const int tag = 1337;
            int rc = MPI_Send(static_cast<void *>(areaTotals.data()), 4,
                              MPI_DOUBLE, 0, tag, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
        }

        int rc = MPI_Bcast(static_cast<void *>(areaTotals.data()), 4,
                           MPI_DOUBLE, 0, params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        setAreaTotals(areaTotals);
    }

    launchPostIntegrate(params, ip);

    if (ip.useGasExchange) {
        assert(nullptr != ip.hNumToBeDeleted && "Given pointer is nullptr");
        getNumToBeDeleted(static_cast<void *>(ip.hNumToBeDeleted), sizeof(int),
                          true);
    }

    void *memStart = static_cast<void *>(params.maximums.data());
    CUDA_CALL(cudaMemcpy(memStart, static_cast<void *>(params.blockMax),
                         3 * GRID_SIZE * sizeof(double), cudaMemcpyDefault));

    ip.maxRadius = 0.0;
    ip.maxExpansion = 0.0;
    ip.maxError = 0.0;
    uint32_t n = 1;
    if (params.bubbles.count > BLOCK_SIZE) {
        float temp = static_cast<float>(params.bubbles.count) / BLOCK_SIZE;
        n = static_cast<uint32_t>(std::ceil(temp));
        n = std::min(n, static_cast<uint32_t>(GRID_SIZE));
    }
    double *p = static_cast<double *>(memStart);
    for (uint32_t i = 0; i < n; i++) {
        ip.maxError = std::max(ip.maxError, p[i]);
        ip.maxRadius = std::max(ip.maxRadius, p[i + GRID_SIZE]);
        ip.maxExpansion = std::max(ip.maxExpansion, p[i + 2 * GRID_SIZE]);
    }

    std::array<double, 5> maximums;
    if (params.nProcs > 1) {
        getTotalVolumeNew(&maximums[0]);
        maximums[1] = ip.maxError;
        maximums[2] = ip.maxRadius;
        maximums[3] = ip.maxExpansion;

        if (params.rank == 0) {
            std::array<double, 4> temp;
            for (int i = 1; i < params.nProcs; i++) {
                const int tag = 1337;
                MPI_Status status;
                int rc = MPI_Recv(static_cast<void *>(temp.data()), 4,
                                  MPI_DOUBLE, i, tag, params.comm, &status);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
                maximums[0] += temp[0];
                maximums[1] = std::max(maximums[1], temp[1]);
                maximums[2] = std::max(maximums[2], temp[2]);
                maximums[3] = std::max(maximums[3], temp[3]);
            }
        } else {
            const int tag = 1337;
            int rc = MPI_Send(static_cast<void *>(maximums.data()), 4,
                              MPI_DOUBLE, 0, tag, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
        }
    }

    auto calculateNewTimestep = [&params, &ip]() {
        double &ts = params.hostData.timeStep;
        ip.errorTooLarge = ip.maxError > params.hostData.errorTolerance;
        const bool increaseTs =
            ip.maxError < 0.45 * params.hostData.errorTolerance && ts < 10;
        if (ip.errorTooLarge) {
            ts *= 0.37;
        } else if (increaseTs) {
            ts *= 1.269;
        }
    };

    if (params.nProcs > 1) {
        if (params.rank == 0) {
            setTotalVolumeNew(&maximums[0]);
            ip.maxError = maximums[1];
            ip.maxRadius = maximums[2];
            ip.maxExpansion = maximums[3];

            calculateNewTimestep();

            maximums[4] = params.hostData.timeStep;
        }

        // Bcast
        int rc = MPI_Bcast(static_cast<void *>(maximums.data()), 5, MPI_DOUBLE,
                           0, params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        if (params.rank != 0) {
            setTotalVolumeNew(&maximums[0]);
            ip.maxError = maximums[1];
            ip.maxRadius = maximums[2];
            ip.maxExpansion = maximums[3];
            ip.errorTooLarge = ip.maxError > params.hostData.errorTolerance;
            params.hostData.timeStep = maximums[4];
        }
    } else {
        calculateNewTimestep();
    }

    nvtxRangePop();
}

void integrate(Params &params, IntegrationParams &ip) {
    nvtxRangePush("Intergration");

    do {
        step(params, ip);
    } while (ip.errorTooLarge);

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

    if (ip.useGasExchange) {
        swapper = params.bubbles.r;
        params.bubbles.r = params.bubbles.rp;
        params.bubbles.rp = swapper;

        swapper = params.bubbles.drdto;
        params.bubbles.drdto = params.bubbles.drdt;
        params.bubbles.drdt = params.bubbles.drdtp;
        params.bubbles.drdtp = swapper;
    }

    if (ip.incrementPath) {
        swapper = params.bubbles.path;
        params.bubbles.path = params.bubbles.pathNew;
        params.bubbles.pathNew = swapper;
    }

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

    if (ip.useGasExchange) {
        params.hostData.maxBubbleRadius = ip.maxRadius;
        if (*(ip.hNumToBeDeleted) > 0) {
            removeBubbles(params, *(ip.hNumToBeDeleted));
        }
    }

    if (ip.maxExpansion >= 0.5 * params.hostConstants.skinRadius) {
        searchNeighbors(params);
        if (false == ip.useGasExchange) {
            // After search r is correct, but rp is trash.
            // pairwiseInteraction always uses predicted values, so copy r to rp
            uint64_t bytes = params.bubbles.stride * sizeof(double);
            CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                                      static_cast<void *>(params.bubbles.r),
                                      bytes, cudaMemcpyDefault, 0));
        }
    }

    nvtxRangePop();
}

void stabilize(Params &params, int numStepsToRelax) {
    nvtxRangePush("Stabilization");
    params.hostData.energy1 = totalEnergy(params);

    IntegrationParams ip;
    ip.useGasExchange = false;
    ip.useFlow = false;
    ip.incrementPath = false;
    ip.errorTooLarge = true;
    ip.maxRadius = 0.0;
    ip.maxExpansion = 0.0;
    ip.maxError = 0.0;

    nvtxRangePush("For-loop");
    for (int i = 0; i < numStepsToRelax; ++i) {
        integrate(params, ip);
    }
    nvtxRangePop();

    params.hostData.energy2 = totalEnergy(params);
    nvtxRangePop();
}

double boxVolume(Params &params) {
    dvec temp = params.hostConstants.globalInterval;
    return (params.hostConstants.dimensionality == 3) ? temp.x * temp.y * temp.z
                                                      : temp.x * temp.y;
}

void saveSnapshot(Params &params) {
    // Calculate energies of bubbles to tempD1, but don't reduce.
    launchResetArray(params, params.bubbles.count, false, 0.0, params.tempD1);
    launchPotentialEnergy(params);
    // Make sure the thread is not working
    if (params.ioThread.joinable()) {
        params.ioThread.join();
    }

    // Copy all device memory to host.
    void *memStart = static_cast<void *>(params.hostMemory.data());
    CUDA_CALL(
        cudaMemcpyAsync(memStart, params.memory,
                        params.hostMemory.size() * sizeof(params.hostMemory[0]),
                        cudaMemcpyDefault, 0));
    CUDA_CALL(cudaEventRecord(params.snapshotParams.event, 0));

    // This lambda helps calculate the host address for each pointer from the
    // device address.
    auto getHostPtr = [&params, &memStart](auto devPtr) -> decltype(devPtr) {
        return static_cast<decltype(devPtr)>(memStart) +
               (devPtr - static_cast<decltype(devPtr)>(params.memory));
    };

    // Get the host pointers from the device pointers
    params.snapshotParams.x = getHostPtr(params.bubbles.x);
    params.snapshotParams.y = getHostPtr(params.bubbles.y);
    params.snapshotParams.z = getHostPtr(params.bubbles.z);
    params.snapshotParams.r = getHostPtr(params.bubbles.r);
    params.snapshotParams.vx = getHostPtr(params.bubbles.dxdt);
    params.snapshotParams.vy = getHostPtr(params.bubbles.dydt);
    params.snapshotParams.vz = getHostPtr(params.bubbles.dzdt);
    params.snapshotParams.vr = getHostPtr(params.bubbles.drdt);
    params.snapshotParams.path = getHostPtr(params.bubbles.path);
    params.snapshotParams.error = getHostPtr(params.bubbles.error);
    params.snapshotParams.energy = getHostPtr(params.tempD1);
    params.snapshotParams.index = getHostPtr(params.bubbles.index);
    params.snapshotParams.wrapCountX = getHostPtr(params.bubbles.wrapCountX);
    params.snapshotParams.wrapCountY = getHostPtr(params.bubbles.wrapCountY);
    params.snapshotParams.wrapCountZ = getHostPtr(params.bubbles.wrapCountZ);

    params.snapshotParams.count = params.bubbles.count;

    auto writeSnapshot = [](const SnapshotParams &snapshotParams,
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
                const int ind = snapshotParams.index[i];
                const double xi = snapshotParams.x[i];
                const double yi = snapshotParams.y[i];
                const double zi = snapshotParams.z[i];
                const double vxi = snapshotParams.vx[i];
                const double vyi = snapshotParams.vy[i];
                const double vzi = snapshotParams.vz[i];
                const double px = xPrev[ind];
                const double py = yPrev[ind];
                const double pz = zPrev[ind];

                double displX = abs(xi - px);
                displX = displX > 0.5 * snapshotParams.interval.x
                             ? displX - snapshotParams.interval.x
                             : displX;
                double displY = abs(yi - py);
                displY = displY > 0.5 * snapshotParams.interval.y
                             ? displY - snapshotParams.interval.y
                             : displY;
                double displZ = abs(zi - pz);
                displZ = displZ > 0.5 * snapshotParams.interval.z
                             ? displZ - snapshotParams.interval.z
                             : displZ;

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
                file << sqrt(displX * displX + displY * displY +
                             displZ * displZ);
                file << ",";
                file << snapshotParams.error[i];
                file << ",";
                file << ind + snapshotParams.indexOffset;
                file << "\n";

                xPrev[ind] = xi;
                yPrev[ind] = yi;
                zPrev[ind] = zi;
            }
        }
    };

    // Spawn a new thread to write the snapshot to a file
    params.ioThread =
        std::thread(writeSnapshot, std::cref(params.snapshotParams),
                    params.hostData.numSnapshots++, params.previousX.data(),
                    params.previousY.data(), params.previousZ.data());
}

void end(Params &params) {
    if (params.rank == 0) {
        printf("Cleaning up...\n");
    }
    if (params.ioThread.joinable()) {
        params.ioThread.join();
    }

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(params.deviceConstants)));
    CUDA_CALL(cudaFree(params.memory));
    CUDA_CALL(cudaFreeHost(static_cast<void *>(params.pinnedMemory)));
    CUDA_CALL(cudaFree(params.outgoingBubbles.memory));
    CUDA_CALL(cudaFree(params.incomingBubbles.memory));
}

void init(const char *inputFileName, Params &params) {
    if (params.rank == 0) {
        printf("==============\nInitialization\n==============\n");
        printf("Reading inputs from %s\n", inputFileName);
    }
    nlohmann::json inputJson;
    std::fstream file(inputFileName, std::ios::in);
    if (file.is_open()) {
        file >> inputJson;
    } else {
        throw std::runtime_error("Couldn't open input file!");
    }

    auto constants = inputJson["constants"];
    auto bubbles = inputJson["bubbles"];
    auto box = inputJson["box"];
    auto wall = box["wall"];
    auto flow = inputJson["flow"];

    const int stabilizationSteps = inputJson["stabilization"]["steps"];

    params.hostData.avgRad = bubbles["radius"]["mean"];
    params.hostData.minNumBubbles = bubbles["numEnd"];
    params.hostConstants.skinRadius *= params.hostData.avgRad;

    const double mu = constants["mu"]["value"];
    const double phi = constants["phi"]["value"];
    params.hostConstants.minRad = 0.1 * params.hostData.avgRad;
    params.hostConstants.fZeroPerMuZero =
        (float)constants["sigma"]["value"] * params.hostData.avgRad / mu;
    params.hostConstants.kParameter = constants["K"]["value"];
    params.hostConstants.kappa = constants["kappa"]["value"];
    params.hostData.timeScalingFactor =
        params.hostConstants.kParameter /
        (params.hostData.avgRad * params.hostData.avgRad);

    params.hostData.addFlow = 1 == flow["impose"];
    params.hostConstants.flowLbb = flow["lbb"];
    params.hostConstants.flowTfr = flow["tfr"];
    params.hostConstants.flowVel = flow["velocity"];
    params.hostConstants.flowVel *= params.hostConstants.fZeroPerMuZero;

    params.hostData.errorTolerance = inputJson["errorTolerance"]["value"];
    params.hostData.snapshotFrequency = inputJson["snapShot"]["frequency"];
    std::stringstream ss;
    ss << inputJson["snapShot"]["filename"];
    ss << "r" << params.rank;
    params.snapshotParams.name = ss.str();

    params.hostConstants.wallDragStrength = wall["drag"];
    params.hostConstants.xWall = 1 == wall["x"];
    params.hostConstants.yWall = 1 == wall["y"];
    params.hostConstants.zWall = 1 == wall["z"];
    params.hostConstants.dimensionality = box["dimensionality"];

    // Calculate the size of the global simulation box
    auto computeGlobalBox = [&bubbles, &box, &params]() {
        const float d = 2 * params.hostData.avgRad;
        float n = (float)bubbles["numStart"];
        dvec dimensions = box["relativeDimensions"];

        if (params.hostConstants.dimensionality == 3) {
            n = std::cbrt(n);
            const float a = std::cbrt(dimensions.x / dimensions.y);
            const float b = std::cbrt(dimensions.x / dimensions.z);
            const float c = std::cbrt(dimensions.y / dimensions.z);
            dimensions = dvec(a * b, c / a, 1.0 / (b * c));
        } else {
            n = std::sqrt(n);
            const float a = std::sqrt(dimensions.x / dimensions.y);
            dimensions = dvec(a, 1.0 / a, 0.0);
        }

        params.hostConstants.globalInterval = (n * d * dimensions).getCeil();
        params.hostConstants.globalTfr = params.hostConstants.globalInterval;
        params.hostConstants.globalLbb = dvec(0, 0, 0);
    };

    auto computeLocalDimensions = [&params](int nProcs, int rank) {
        // Calculate the local dimensions from the global using the rank
        dvec &tfr = params.hostConstants.tfr;
        dvec &lbb = params.hostConstants.lbb;
        dvec &interval = params.hostConstants.interval;
        const dvec &gi = params.hostConstants.globalInterval;

        if (1 < nProcs) {
            // What follows is a dirty and hacky way to decompose the domain

            const bool twoProcs = nProcs == 2;
            const bool uneven = (nProcs & 1) == 0x1;
            const bool twoByTwo = params.hostConstants.dimensionality == 2 &&
                                  gi.x == gi.y && nProcs == 4;
            const bool twoByTwoByTwo =
                params.hostConstants.dimensionality == 3 && gi.x == gi.y &&
                gi.x == gi.z && nProcs == 8;

            auto divideEvenly = [&tfr, &lbb, &params, &gi, &rank, &nProcs]() {
                // Just divide the largest dimension equally to the GPUs
                tfr = params.hostConstants.globalTfr;
                lbb = params.hostConstants.globalLbb;

                if (gi.x >= gi.y && gi.x >= gi.z) {
                    tfr.x = gi.x / nProcs * (rank + 1);
                    lbb.x = gi.x / nProcs * rank;
                } else if (gi.y >= gi.x && gi.y >= gi.z) {
                    tfr.y = gi.y / nProcs * (rank + 1);
                    lbb.y = gi.y / nProcs * rank;
                } else {
                    tfr.z = gi.z / nProcs * (rank + 1);
                    lbb.z = gi.z / nProcs * rank;
                }
            };

            auto middleInHalfMaxInNPer2Parts = [&tfr, &lbb, &params, &gi, &rank,
                                                &nProcs]() {
                // Divide the middle dimension in half and the
                // larger into nprocs/2 parts
                tfr = params.hostConstants.globalTfr;
                lbb = params.hostConstants.globalLbb;

                const int n = nProcs / 2;
                const int temp = rank % n;
                const int temp2 = rank / n;

                if (gi.x >= gi.y && gi.x >= gi.z) {
                    tfr.x = gi.x / n * (temp + 1);
                    lbb.x = gi.x / n * temp;
                    if (gi.y >= gi.z) {
                        tfr.y = gi.y / 2.0 * (temp2 + 1);
                        lbb.y = gi.y / 2.0 * temp2;
                    } else {
                        tfr.z = gi.z / 2.0 * (temp2 + 1);
                        lbb.z = gi.z / 2.0 * temp2;
                    }
                } else if (gi.y >= gi.x && gi.y >= gi.z) {
                    tfr.y = gi.y / n * (temp + 1);
                    lbb.y = gi.y / n * temp;
                    if (gi.x >= gi.z) {
                        tfr.x = gi.x / 2.0 * (temp2 + 1);
                        lbb.x = gi.x / 2.0 * temp2;
                    } else {
                        tfr.z = gi.z / 2.0 * (temp2 + 1);
                        lbb.z = gi.z / 2.0 * temp2;
                    }
                } else {
                    tfr.z = gi.z / n * (temp + 1);
                    lbb.z = gi.z / n * temp;
                    if (gi.x >= gi.y) {
                        tfr.x = gi.x / 2.0 * (temp2 + 1);
                        lbb.x = gi.x / 2.0 * temp2;
                    } else {
                        tfr.y = gi.y / 2.0 * (temp2 + 1);
                        lbb.y = gi.y / 2.0 * temp2;
                    }
                }
            };

            auto cubeIntoEightEvenParts = [&tfr, &lbb, &params, &gi, &rank,
                                           &nProcs]() {
                const int xi = rank & 0x1;
                const int yi =
                    rank == 2 || rank == 3 || rank == 6 || rank == 7 ? 1 : 0;
                const int zi = rank / 4;
                const ivec temp = ivec(xi, yi, zi);
                tfr = gi / 2.0 * (temp + 1).asType<double>();
                lbb = gi / 2.0 * temp.asType<double>();
            };

            if (twoProcs || uneven) {
                divideEvenly();
            } else if (twoByTwo) {
                const ivec temp = ivec(rank & 0x1, rank / 2, 0);
                tfr = gi / 2.0 * (temp + 1).asType<double>();
                lbb = gi / 2.0 * temp.asType<double>();
            } else if (twoByTwoByTwo) {
                cubeIntoEightEvenParts();
            } else {
                if (3 == params.hostConstants.dimensionality) {
                    if (gi.getMinComponent() / gi.getMidComponent() < 0.3) {
                        // The smallest dimension is less than third of the
                        // second smallest. Treat the box essentially as two
                        // dimensional.
                        if (gi.getMidComponent() / gi.getMaxComponent() < 0.3) {
                            // The box is long, but thin in two dimensions
                            divideEvenly();
                        } else {
                            middleInHalfMaxInNPer2Parts();
                        }
                    } else {
                        // Minimum is at least 1/3 of the middle dimension
                        if (gi.getMidComponent() / gi.getMaxComponent() < 0.3) {
                            divideEvenly();
                        } else {
                            if (nProcs == 8) {
                                cubeIntoEightEvenParts();
                            } else {
                                middleInHalfMaxInNPer2Parts();
                            }
                        }
                    }
                } else {
                    if (gi.getMidComponent() / gi.getMaxComponent() < 0.3) {
                        divideEvenly();
                    } else {
                        middleInHalfMaxInNPer2Parts();
                    }
                }
            }
        } else {
            tfr = params.hostConstants.globalTfr;
            lbb = params.hostConstants.globalLbb;
        }

        interval = tfr - lbb;
    };

    auto findNeighborProcessors = [&params]() {
        const dvec &tfr = params.hostConstants.tfr;
        const dvec &lbb = params.hostConstants.lbb;

        std::vector<dvec> lbbs(params.nProcs);
        std::vector<dvec> tfrs(params.nProcs);

        std::vector<double> data(params.nProcs * 6);
        data[params.rank * 6 + 0] = lbb.x;
        data[params.rank * 6 + 1] = lbb.y;
        data[params.rank * 6 + 2] = lbb.z;
        data[params.rank * 6 + 3] = tfr.x;
        data[params.rank * 6 + 4] = tfr.y;
        data[params.rank * 6 + 5] = tfr.z;

        for (int i = 0; i < params.nProcs; i++) {
            int rc = MPI_Bcast(static_cast<void *>(&data[6 * i]), 6, MPI_DOUBLE,
                               i, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
            lbbs[i] = dvec(data[6 * i + 0], data[6 * i + 1], data[6 * i + 2]);
            tfrs[i] = dvec(data[6 * i + 3], data[6 * i + 4], data[6 * i + 5]);
        }

        auto getProcNum = [&lbbs, &tfrs, &params](dvec &&p) {
            const ivec &gi = params.hostConstants.globalInterval.asType<int>();
            ivec temp = p.asType<int>() + gi;
            p.x = static_cast<double>(temp.x % gi.x);
            p.y = static_cast<double>(temp.y % gi.y);
            if (3 == params.hostConstants.dimensionality) {
                p.z = static_cast<double>(temp.z % gi.z);
            }

            for (int i = 0; i < params.nProcs; i++) {
                bool inside = p.x > lbbs[i].x && p.x < tfrs[i].x;
                inside &= p.y > lbbs[i].y && p.y < tfrs[i].y;
                if (3 == params.hostConstants.dimensionality) {
                    inside &= p.z > lbbs[i].z && p.z < tfrs[i].z;
                }
                if (inside) {
                    return i;
                }
            }

            return -1;
        };

        if (3 == params.hostConstants.dimensionality) {
            params.areaToProcessorMap[0] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y + 1.0, lbb.z + 1.0));
            params.areaToProcessorMap[1] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y + 1.0, lbb.z + 1.0));

            params.areaToProcessorMap[2] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y - 1.0, lbb.z + 1.0));
            params.areaToProcessorMap[3] =
                getProcNum(dvec(lbb.x + 1.0, tfr.y + 1.0, lbb.z + 1.0));

            params.areaToProcessorMap[4] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y + 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[5] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y + 1.0, tfr.z + 1.0));

            // x is free
            params.areaToProcessorMap[6] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y - 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[7] =
                getProcNum(dvec(lbb.x + 1.0, tfr.y + 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[8] =
                getProcNum(dvec(lbb.x + 1.0, tfr.y + 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[9] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y - 1.0, tfr.z + 1.0));

            // y is free
            params.areaToProcessorMap[10] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y + 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[11] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y + 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[12] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y + 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[13] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y + 1.0, lbb.z - 1.0));

            // z is free
            params.areaToProcessorMap[14] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y - 1.0, lbb.z + 1.0));
            params.areaToProcessorMap[15] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y - 1.0, lbb.z + 1.0));
            params.areaToProcessorMap[16] =
                getProcNum(dvec(tfr.x + 1.0, tfr.y + 1.0, lbb.z + 1.0));
            params.areaToProcessorMap[17] =
                getProcNum(dvec(lbb.x - 1.0, tfr.y + 1.0, lbb.z + 1.0));

            // eight corners
            params.areaToProcessorMap[18] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y - 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[19] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y - 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[20] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y - 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[21] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y - 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[22] =
                getProcNum(dvec(tfr.x + 1.0, tfr.y + 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[23] =
                getProcNum(dvec(lbb.x - 1.0, tfr.y + 1.0, lbb.z - 1.0));
            params.areaToProcessorMap[24] =
                getProcNum(dvec(lbb.x - 1.0, tfr.y + 1.0, tfr.z + 1.0));
            params.areaToProcessorMap[25] =
                getProcNum(dvec(tfr.x + 1.0, tfr.y + 1.0, tfr.z + 1.0));
        } else {
            // y is free
            params.areaToProcessorMap[0] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y + 1.0, 0.0));
            params.areaToProcessorMap[1] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y + 1.0, 0.0));

            // x is free
            params.areaToProcessorMap[2] =
                getProcNum(dvec(lbb.x + 1.0, lbb.y - 1.0, 0.0));
            params.areaToProcessorMap[3] =
                getProcNum(dvec(lbb.x + 1.0, tfr.y + 1.0, 0.0));

            // four corners
            params.areaToProcessorMap[4] =
                getProcNum(dvec(lbb.x - 1.0, lbb.y - 1.0, 0.0));
            params.areaToProcessorMap[5] =
                getProcNum(dvec(tfr.x + 1.0, lbb.y - 1.0, 0.0));
            params.areaToProcessorMap[6] =
                getProcNum(dvec(tfr.x + 1.0, tfr.y + 1.0, 0.0));
            params.areaToProcessorMap[7] =
                getProcNum(dvec(lbb.x - 1.0, tfr.y + 1.0, 0.0));
            params.areaToProcessorMap[8] = 0;
            params.areaToProcessorMap[9] = 0;
            params.areaToProcessorMap[10] = 0;
            params.areaToProcessorMap[11] = 0;
            params.areaToProcessorMap[12] = 0;
            params.areaToProcessorMap[13] = 0;
            params.areaToProcessorMap[14] = 0;
            params.areaToProcessorMap[15] = 0;
            params.areaToProcessorMap[16] = 0;
            params.areaToProcessorMap[17] = 0;
            params.areaToProcessorMap[18] = 0;
            params.areaToProcessorMap[19] = 0;
            params.areaToProcessorMap[20] = 0;
            params.areaToProcessorMap[21] = 0;
            params.areaToProcessorMap[22] = 0;
            params.areaToProcessorMap[23] = 0;
            params.areaToProcessorMap[24] = 0;
            params.areaToProcessorMap[25] = 0;
        }
    };

    printf("Computing global box\n");
    computeGlobalBox();
    printf("Computing local box\n");
    computeLocalDimensions(params.nProcs, params.rank);
    if (1 < params.nProcs) {
        printf("Finding neigboring processors\n");
        findNeighborProcessors();
    }

#ifndef NDEBUG
    ss.clear();
    ss.str(std::string());
    ss << "debug" << params.rank << ".dat";
    std::ofstream debugFile(ss.str().c_str(), std::ios_base::app);
    if (debugFile.is_open()) {
        const dvec &tfr = params.hostConstants.tfr;
        const dvec &lbb = params.hostConstants.lbb;
        const dvec &gtfr = params.hostConstants.globalTfr;
        const dvec &glbb = params.hostConstants.globalLbb;
        debugFile << params.rank << "\n"
                  << lbb.x << "," << lbb.y << "," << lbb.z << "\n"
                  << tfr.x << "," << tfr.y << "," << tfr.z << "\n"
                  << glbb.x << "," << glbb.y << "," << glbb.z << "\n"
                  << gtfr.x << "," << gtfr.y << "," << gtfr.z << "\n";
        for (const auto &it : params.areaToProcessorMap) {
            debugFile << it << ",";
        }
        debugFile << "\n";
        debugFile << std::flush;
    } else {
        printf("Couldn't open file stream to append debug info "
               "to!\n");
    }

    int i = 0;
    for (const auto &it : params.areaToProcessorMap) {
        if (it == -1) {
            printf("A bad value in areaToProcessorMap: %d, %d\n", it, i);
            assert(false);
        }
        i++;
    }
#endif

    setAreaToProcessorMap(static_cast<void *>(params.areaToProcessorMap.data()),
                          sizeof(int) * params.areaToProcessorMap.size());

    // Local count
    ivec bubblesPerDim =
        (params.hostConstants.interval / (2 * params.hostData.avgRad))
            .getCeil()
            .asType<int>();
    params.bubbles.count = bubblesPerDim.x * bubblesPerDim.y;
    if (params.hostConstants.dimensionality == 3) {
        params.bubbles.count *= bubblesPerDim.z;
    }

    // If one wants to keep the static indices global, one should add
    // communication here between the processes to send the total number of
    // bubbles each has and then based on the rank determine which bubbles are
    // before the bubbles of this rank
    params.snapshotParams.indexOffset = 0;

    // Calculate the length of 'rows'.
    // Make it divisible by 32, as that's the warp size.
    params.bubbles.stride =
        params.bubbles.count +
        !!(params.bubbles.count % 32) * (32 - params.bubbles.count % 32);

    // It seems to hold that in 3 dimensions the total number of
    // bubble pairs is 11x and in two dimensions 4x number of bubbles.
    // Note that these numbers depend on the "skin radius", i.e.
    // from how far are the neighbors looked for.
    const uint32_t avgNumNeighbors =
        (params.hostConstants.dimensionality == 3) ? 12 : 4;
    params.pairs.stride = avgNumNeighbors * params.bubbles.stride;

    if (params.rank == 0) {
        printf("---------------Starting parameters---------------\n");
        params.hostConstants.print();
        params.hostData.print();
        params.bubbles.print();
        params.pairs.print();
        printf("-------------------------------------------------\n");
    }

    // Allocate and copy constants to GPU
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceConstants),
                           sizeof(Constants)));
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceConstants),
                         static_cast<void *>(&params.hostConstants),
                         sizeof(Constants), cudaMemcpyDefault));
    setConstants(static_cast<void *>(&params.deviceConstants),
                 sizeof(Constants *));

    CUDA_CALL(
        cudaEventCreate(&params.snapshotParams.event, cudaEventDisableTiming));
    printRelevantInfoOfCurrentDevice();

    // Set device globals to zero
    setNumPairsToZero();
    launchInitGlobals(params);

    if (params.rank == 0) {
        printf("Reserving device memory\n");
    }
    CUDA_CALL(cudaMallocHost(&params.pinnedMemory, sizeof(int)));

    // Total memory: memory for bubble data, memory for pair data and memory for
    // temporary arrays
    uint64_t bytes = params.bubbles.getMemReq();
    bytes += params.pairs.getMemReq();
    bytes += params.getTempMemReq();
    CUDA_ASSERT(cudaMalloc(&params.memory, bytes));

    params.maximums.resize(3 * GRID_SIZE);

    // If we're going to be saving snapshots, allocate enough memory to hold all
    // the device data.
    if (0.0 < params.hostData.snapshotFrequency) {
        params.hostMemory.resize(bytes);
    }

    // Each named pointer is setup by these functions to point to
    // a different stride inside the continuous memory blob
    void *pairStart = params.bubbles.setupPointers(params.memory);
    pairStart = params.pairs.setupPointers(pairStart);
    params.setTempPointers(pairStart);

    params.previousX.resize(params.bubbles.stride);
    params.previousY.resize(params.bubbles.stride);
    params.previousZ.resize(params.bubbles.stride);
    params.snapshotParams.x0.resize(params.bubbles.stride);
    params.snapshotParams.y0.resize(params.bubbles.stride);
    params.snapshotParams.z0.resize(params.bubbles.stride);

    const uint64_t megs = bytes / (1024 * 1024);
    const uint64_t kilos = (bytes - megs * 1024 * 1024) / 1024;
    bytes = (bytes - megs * 1024 * 1024 - kilos * 1024);

    if (params.rank == 0) {
        printf("Allocated %ld MB %ld KB %ld B of global device memory.\n", megs,
               kilos, bytes);
        printf("Generating starting data\n");
    }

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(
        generator,
        static_cast<int>(inputJson["rngSeed"]["value"]) + params.rank));
    if (params.hostConstants.dimensionality == 3)
        CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.z,
                                                params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.x,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.y,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateUniformDouble(generator, params.bubbles.rp,
                                            params.bubbles.count));
    CURAND_CALL(curandGenerateNormalDouble(
        generator, params.bubbles.r, params.bubbles.count,
        params.hostData.avgRad, bubbles["radius"]["std"]));
    CURAND_CALL(curandDestroyGenerator(generator));

    launchAssignDataToBubbles(params, bubblesPerDim);

    // Get the average input surface area and maximum bubble radius
    void *cubPtr = static_cast<void *>(params.tempPair2);
    void *out = static_cast<void *>(&params.hostConstants.averageSurfaceAreaIn);
    cubSum(cubPtr, params.pairs.getMemReq() / 2, params.bubbles.rp, out,
           params.bubbles.count, (cudaStream_t)0, false);

    out = static_cast<void *>(&params.hostData.maxBubbleRadius);
    cubMax(cubPtr, params.pairs.getMemReq() / 2, params.bubbles.r, out,
           params.bubbles.count, (cudaStream_t)0, false);

    if (1 < params.nProcs) {
        std::array<double, 3> radArea;
        radArea[0] = params.hostData.maxBubbleRadius;
        radArea[1] = params.hostConstants.averageSurfaceAreaIn;
        radArea[2] = (double)params.bubbles.count;

        const int tag = 1337;
        MPI_Status status;
        if (params.rank == 0) {
            std::array<double, 3> temp;
            for (int i = 1; i < params.nProcs; i++) {
                int rc = MPI_Recv(static_cast<void *>(temp.data()), 3,
                                  MPI_DOUBLE, i, tag, params.comm, &status);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
                radArea[0] = std::max(temp[0], radArea[0]);
                radArea[1] += temp[1];
                radArea[2] += temp[2];
            }
            radArea[1] /= radArea[2];
        } else {
            int rc = MPI_Send(static_cast<void *>(radArea.data()), 3,
                              MPI_DOUBLE, 0, tag, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
        }

        int rc = MPI_Bcast(static_cast<void *>(radArea.data()), 3, MPI_DOUBLE,
                           0, params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }

        params.hostData.maxBubbleRadius = radArea[0];
        params.hostConstants.averageSurfaceAreaIn = radArea[1];
    } else {
        params.hostConstants.averageSurfaceAreaIn /= params.bubbles.count;
    }

    if (params.rank == 0) {
        printf("First neighbor search\n");
    }
    searchNeighbors(params);

    // After search x, y, z, r are correct, but all predicted are trash.
    // pairwiseInteraction always uses predicted values, so copy currents to
    // predicteds
    bytes = params.bubbles.stride * sizeof(double);
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

    if (params.rank == 0) {
        printf("Calculating initial velocities for Adams-Bashforth-Moulton\n");
    }
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dxdto);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dydto);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dzdto);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.drdto);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dxdtp);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dydtp);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.dzdtp);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.drdtp);
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.path);

    const uint32_t dynSharedMemBytes =
        params.hostConstants.dimensionality * BLOCK_SIZE * sizeof(double);

    IntegrationParams ip;
    ip.useGasExchange = false;
    ip.useFlow = false;

    launchPairwiseInteraction(params, ip, dynSharedMemBytes);
    launchEuler(params);

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

    launchPairwiseInteraction(params, ip, dynSharedMemBytes);

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

    if (params.rank == 0) {
        printf("Stabilizing a few rounds after creation\n");
    }
    for (uint32_t i = 0; i < 5; ++i) {
        stabilize(params, stabilizationSteps);
#ifndef NDEBUG
        // Just for testing.
        if (params.rank == 0) {
            printf("Quitting after first stabilization.\n");
        }
        saveSnapshot(params);
        return;
#endif
    }

    if (params.rank == 0) {
        printf("Scaling the simulation box\n");
    }
    double bubbleVolume = totalVolume(params);
    if (1 < params.nProcs) {
        const int tag = 1337;
        MPI_Status status;
        if (params.rank == 0) {
            double temp = 0;
            for (int i = 1; i < params.nProcs; i++) {
                int rc = MPI_Recv(static_cast<void *>(&temp), 1, MPI_DOUBLE, i,
                                  tag, params.comm, &status);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
                bubbleVolume += temp;
            }
        } else {
            int rc = MPI_Send(static_cast<void *>(&bubbleVolume), 1, MPI_DOUBLE,
                              0, tag, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
        }

        int rc = MPI_Bcast(static_cast<void *>(&bubbleVolume), 1, MPI_DOUBLE, 0,
                           params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }
    }

    if (params.rank == 0) {
        printf("Current phi: %.9g, target phi: %.9g\n",
               bubbleVolume / boxVolume(params), phi);
    }

    launchTransformPositions(params, true);

    dvec relDim = box["relativeDimensions"];
    double t = bubbleVolume / (phi * relDim.x * relDim.y);
    if (params.hostConstants.dimensionality == 3) {
        t /= relDim.z;
        t = std::cbrt(t);
    } else {
        t = std::sqrt(t);
        relDim.z = 0.0;
    }

    params.hostConstants.globalTfr = dvec(t, t, t) * relDim;
    params.hostConstants.globalInterval =
        params.hostConstants.globalTfr - params.hostConstants.globalLbb;
    params.hostConstants.flowTfr =
        params.hostConstants.globalInterval * params.hostConstants.flowTfr +
        params.hostConstants.globalLbb;
    params.hostConstants.flowLbb =
        params.hostConstants.globalInterval * params.hostConstants.flowLbb +
        params.hostConstants.globalLbb;
    params.snapshotParams.interval = params.hostConstants.globalInterval;

    // Update local dimensions
    computeLocalDimensions(params.nProcs, params.rank);

    double mult = phi * boxVolume(params) / CUBBLE_PI;
    if (params.hostConstants.dimensionality == 3) {
        mult = std::cbrt(0.75 * mult);
    } else {
        mult = std::sqrt(mult);
    }
    params.hostConstants.bubbleVolumeMultiplier = mult;

    // Copy the updated constants to GPU
    CUDA_CALL(cudaMemcpy(static_cast<void *>(params.deviceConstants),
                         static_cast<void *>(&params.hostConstants),
                         sizeof(Constants), cudaMemcpyDefault));

    launchTransformPositions(params, false);

    if (params.rank == 0) {
        printf("Current phi: %.9g, target phi: %.9g\n",
               bubbleVolume / boxVolume(params), phi);
    }

    if (params.rank == 0) {
        printf("Neighbor search after scaling\n");
    }

    searchNeighbors(params);
    // After search r is correct, but rp is trash.
    // pairwiseInteraction always uses predicted values, so copy r to rp
    bytes = params.bubbles.stride * sizeof(double);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.bubbles.rp),
                              static_cast<void *>(params.bubbles.r), bytes,
                              cudaMemcpyDefault, 0));

    if (params.rank == 0) {
        printf("Stabilizing a few rounds after scaling\n");
    }
    for (uint32_t i = 0; i < 5; ++i)
        stabilize(params, stabilizationSteps);

    if (params.rank == 0) {
        printf("\n=============\nStabilization\n=============\n");
    }
    params.hostData.numNeighborsSearched = 0;
    int numSteps = 0;
    const int failsafe = 500;

    if (params.rank == 0) {
        printf("%-7s %-11s %-11s %-11s %-9s\n", "#steps", "dE", "e1", "e2",
               "#searches");
    }
    const double &e1 = params.hostData.energy1;
    const double &e2 = params.hostData.energy2;
    bool stop = false;
    while (false == stop) {
        params.hostData.timeInteger = 0;
        params.hostData.timeFraction = 0.0;

        stabilize(params, stabilizationSteps);

        if (0 == params.rank) {
            double time = ((double)params.hostData.timeInteger +
                           params.hostData.timeFraction) *
                          params.hostData.timeScalingFactor;

            double de = std::abs(e2 - e1);
            if (de > 0.0) {
                de *= 2.0 / ((e2 + e1) * time);
            }

            stop = de < inputJson["stabilization"]["maxDeltaEnergy"] ||
                   (e2 < 1.0 && de < 0.1);
            if (stop) {
                printf("Final energies:");
                printf("\nbefore: %9.5e", e1);
                printf("\nafter: %9.5e", e2);
                printf("\ndelta: %9.5e", de);
                printf("\ntime: %9.5g\n", time);
            } else if (numSteps > failsafe) {
                printf("Over %d steps taken and required delta energy not "
                       "reached. "
                       "Constraints might be too strict.\n",
                       numSteps);
                stop = true;
            } else {
                printf("%-7d ", (numSteps + 1) * stabilizationSteps);
                printf("%-9.5e ", de);
                printf("%-9.5e ", e1);
                printf("%-9.5e ", e2);
                printf("%-9ld\n", params.hostData.numNeighborsSearched);
                params.hostData.numNeighborsSearched = 0;
            }
            ++numSteps;
        }

        int intStop = stop ? 1 : 0;
        int rc = MPI_Bcast(static_cast<void *>(&intStop), 1, MPI_INT, 0,
                           params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }
        stop = (bool)intStop;
    }

    if (0.0 < params.hostData.snapshotFrequency) {
        // Set starting positions.
        // Avoiding batched copy, because the pointers might not be in order
        int *index = reinterpret_cast<int *>(params.hostMemory.data());
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousX.data()),
                             static_cast<void *>(params.bubbles.x),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousY.data()),
                             static_cast<void *>(params.bubbles.y),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(params.previousZ.data()),
                             static_cast<void *>(params.bubbles.z),
                             sizeof(double) * params.bubbles.count,
                             cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(static_cast<void *>(index),
                             static_cast<void *>(params.bubbles.index),
                             sizeof(int) * params.bubbles.count,
                             cudaMemcpyDefault));

        for (int i = 0; i < params.bubbles.count; i++) {
            const int ind = index[i];
            params.snapshotParams.x0[ind] = params.previousX[i];
            params.snapshotParams.y0[ind] = params.previousY[i];
            params.snapshotParams.z0[ind] = params.previousZ[i];
        }

        // 'Previous' vectors are updated with the previous positions after
        // every snapshot, but since there is nothing previous to the first
        // snapshot, initialize them with the starting positions.
        std::copy(params.snapshotParams.x0.begin(),
                  params.snapshotParams.x0.end(), params.previousX.begin());
        std::copy(params.snapshotParams.y0.begin(),
                  params.snapshotParams.y0.end(), params.previousY.begin());
        std::copy(params.snapshotParams.z0.begin(),
                  params.snapshotParams.z0.end(), params.previousZ.begin());
    }

    // Reset wrap counts to 0
    // Avoiding batched memset, because the pointers might not be in order
    bytes = sizeof(int) * params.bubbles.stride;
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountX, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountY, 0, bytes));
    CUDA_CALL(cudaMemset(params.bubbles.wrapCountZ, 0, bytes));

    // Reset errors since integration starts after this
    launchResetArray(params, params.bubbles.count, false, 0.0,
                     params.bubbles.error);

    params.hostData.energy1 = totalEnergy(params);
    if (params.nProcs > 1) {
        const int tag = 1337;
        MPI_Status status;
        if (params.rank == 0) {
            double temp = 0;
            for (int i = 1; i < params.nProcs; i++) {
                int rc = MPI_Recv(static_cast<void *>(&temp), 1, MPI_DOUBLE, i,
                                  tag, params.comm, &status);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
                params.hostData.energy1 += temp;
            }
        } else {
            int rc = MPI_Send(static_cast<void *>(&params.hostData.energy1), 1,
                              MPI_DOUBLE, 0, tag, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }
        }

        int rc = MPI_Bcast(static_cast<void *>(&params.hostData.energy1), 1,
                           MPI_DOUBLE, 0, params.comm);
        if (rc != MPI_SUCCESS) {
            printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                   __LINE__);
        }
    }
    params.hostData.timeInteger = 0;
    params.hostData.timeFraction = 0.0;
    params.hostData.timesPrinted = 1;
    params.hostData.numIntegrationSteps = 0;
}
} // namespace

namespace cubble {
void run(std::string &&inputFileName, int rank, int nProcs) {
    setbuf(stdout, NULL);
    Params params;
    params.rank = rank;
    params.nProcs = nProcs;
    params.comm = MPI_COMM_WORLD;
    init(inputFileName.c_str(), params);

#ifndef NDEBUG
    // This is here just for testing and quitting early
    end(params);
    return;
#endif

    if (params.hostData.snapshotFrequency > 0.0) {
        saveSnapshot(params);
    }

    if (0 == params.rank) {
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
    }

    bool continueSimulation = true;
    double minTimestep = 9999999.9;
    double maxTimestep = -1.0;
    double avgTimestep = 0.0;
    double &ts = params.hostData.timeStep;
    const double minInterval =
        3 == params.hostConstants.dimensionality
            ? 0.5 * params.hostConstants.globalInterval.getMinComponent()
            : 0.5 * (params.hostConstants.globalInterval.x <
                             params.hostConstants.globalInterval.y
                         ? params.hostConstants.globalInterval.x
                         : params.hostConstants.globalInterval.y);

    CUBBLE_PROFILE(true);

    IntegrationParams ip;
    ip.useGasExchange = true;
    ip.useFlow = params.hostData.addFlow;
    ip.incrementPath = true;
    ip.errorTooLarge = true;
    ip.maxRadius = 0.0;
    ip.maxExpansion = 0.0;
    ip.maxError = 0.0;
    ip.hNumToBeDeleted = static_cast<int *>(params.pinnedMemory);

    const double &e1 = params.hostData.energy1;
    const double &e2 = params.hostData.energy2;
    int totalBubbleCount = params.bubbles.count;
    int totalPairCount = params.pairs.count;

    while (continueSimulation) {
        CUBBLE_PROFILE(false);
        integrate(params, ip);

        bool isSnapshotTime = false;
        bool print = false;
        bool resetErrors = false;

        totalBubbleCount = params.bubbles.count;
        totalPairCount = params.pairs.count;
        if (params.nProcs > 1) {
            const int tag = 1337;
            MPI_Status status;

            std::array<int, 2> totals;
            totals[0] = totalBubbleCount;
            totals[1] = totalPairCount;
            if (params.rank == 0) {
                std::array<int, 2> temp;
                for (int i = 1; i < params.nProcs; i++) {
                    int rc = MPI_Recv(static_cast<void *>(temp.data()),
                                      totals.size(), MPI_INT, i, tag,
                                      params.comm, &status);
                    if (rc != MPI_SUCCESS) {
                        printf("Error sendrecving an MPI message at %s:%d\n",
                               __FILE__, __LINE__);
                    }

                    for (uint32_t j = 0; j < totals.size(); j++) {
                        totals[j] += temp[j];
                    }
                }
            } else {
                int rc = MPI_Send(static_cast<void *>(totals.data()),
                                  totals.size(), MPI_INT, 0, tag, params.comm);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }
            }

            int rc = MPI_Bcast(static_cast<void *>(totals.data()),
                               totals.size(), MPI_INT, 0, params.comm);
            if (rc != MPI_SUCCESS) {
                printf("Error sendrecving an MPI message at %s:%d\n", __FILE__,
                       __LINE__);
            }

            totalBubbleCount = totals[0];
            totalPairCount = totals[1];
        }

        // Continue if there are more than the specified minimum number of
        // bubbles left in the simulation and if the largest bubble is
        // smaller than the simulation box in every dimension
        continueSimulation = totalBubbleCount > params.hostData.minNumBubbles &&
                             params.hostData.maxBubbleRadius < minInterval;

        // Track timestep
        minTimestep = ts < minTimestep ? ts : minTimestep;
        maxTimestep = ts > maxTimestep ? ts : maxTimestep;
        avgTimestep += ts;

        // Here we compare potentially very large integers (> 10e6) to each
        // other and small doubles (<= 1.0) to each other to preserve
        // precision.
        const double nextPrintTime =
            params.hostData.timesPrinted / params.hostData.timeScalingFactor;
        const uint64_t nextPrintTimeInteger = (uint64_t)nextPrintTime;
        const double nextPrintTimeFraction =
            nextPrintTime - nextPrintTimeInteger;

        // Print stuff to stdout at the earliest possible moment
        // when simulation time is larger than scaled time
        print = params.hostData.timeInteger > nextPrintTimeInteger ||
                (params.hostData.timeInteger == nextPrintTimeInteger &&
                 params.hostData.timeFraction >= nextPrintTimeFraction);

        if (params.hostData.snapshotFrequency > 0.0) {
            const double nextSnapshotTime = params.hostData.numSnapshots /
                                            (params.hostData.snapshotFrequency *
                                             params.hostData.timeScalingFactor);
            const uint64_t nextSnapshotTimeInteger = (uint64_t)nextSnapshotTime;
            const double nextSnapshotTimeFraction =
                nextSnapshotTime - nextSnapshotTimeInteger;

            isSnapshotTime =
                params.hostData.timeInteger > nextSnapshotTimeInteger ||
                (params.hostData.timeInteger == nextSnapshotTimeInteger &&
                 params.hostData.timeFraction >= nextSnapshotTimeFraction);
        }

        if (print) {
            double energy = totalEnergy(params);
            double bubbleVolume = totalVolume(params);
            double relRad = getSum(params, params.bubbles.r);
            double avgVx = getSum(params, params.bubbles.dxdt);
            double avgVy = getSum(params, params.bubbles.dydt);
            double avgVz = 0.0;
            if (3 == params.hostConstants.dimensionality) {
                avgVz = getSum(params, params.bubbles.dzdt);
            }
            double avgVr = getSum(params, params.bubbles.drdt);
            double avgPath = getSum(params, params.bubbles.path);

            if (params.nProcs > 1) {
                const int tag = 1337;
                MPI_Status status;

                std::array<double, 8> totals;
                totals[0] = energy;
                totals[1] = bubbleVolume;
                totals[2] = relRad;
                totals[3] = avgVx;
                totals[4] = avgVy;
                totals[5] = avgVz;
                totals[6] = avgVr;
                totals[7] = avgPath;
                if (params.rank == 0) {
                    std::array<double, 8> temp;
                    for (int i = 1; i < params.nProcs; i++) {
                        int rc = MPI_Recv(static_cast<void *>(temp.data()),
                                          temp.size(), MPI_DOUBLE, i, tag,
                                          params.comm, &status);
                        if (rc != MPI_SUCCESS) {
                            printf(
                                "Error sendrecving an MPI message at %s:%d\n",
                                __FILE__, __LINE__);
                        }

                        for (uint32_t j = 0; j < totals.size(); j++) {
                            totals[j] += temp[j];
                        }
                    }
                } else {
                    int rc = MPI_Send(static_cast<void *>(totals.data()),
                                      totals.size(), MPI_DOUBLE, 0, tag,
                                      params.comm);
                    if (rc != MPI_SUCCESS) {
                        printf("Error sendrecving an MPI message at %s:%d\n",
                               __FILE__, __LINE__);
                    }
                }

                int rc = MPI_Bcast(static_cast<void *>(totals.data()),
                                   totals.size(), MPI_DOUBLE, 0, params.comm);
                if (rc != MPI_SUCCESS) {
                    printf("Error sendrecving an MPI message at %s:%d\n",
                           __FILE__, __LINE__);
                }

                energy = totals[0];
                bubbleVolume = totals[1];
                relRad = totals[2];
                avgVx = totals[3];
                avgVy = totals[4];
                avgVz = totals[5];
                avgVr = totals[6];
                avgPath = totals[7];
            }

            params.hostData.energy2 = energy;
            if (0 == rank) {
                double de = std::abs(e2 - e1);
                if (de > 0.0) {
                    de *= 2.0 / (e2 + e1);
                }

                relRad /= params.hostData.avgRad * totalBubbleCount;
                avgVx /= totalBubbleCount;
                avgVy /= totalBubbleCount;
                avgVz /= totalBubbleCount;
                avgVr /= totalBubbleCount;
                avgPath /= totalBubbleCount;

                // Add values to data stream
                std::ofstream resultFile("results.dat", std::ios_base::app);
                if (resultFile.is_open()) {
                    resultFile
                        << params.hostData.timesPrinted << " " << relRad << " "
                        << totalBubbleCount << " " << avgPath << " "
                        << params.hostData.energy2 << " " << de << " " << avgVx
                        << " " << avgVy << " " << avgVz << " "
                        << sqrt(avgVx * avgVx + avgVy * avgVy + avgVz * avgVz)
                        << " " << avgVr << "\n";
                } else {
                    printf("Couldn't open file stream to append results "
                           "to!\n");
                }
                const double phi = bubbleVolume / boxVolume(params);
                printf("%-5d ", params.hostData.timesPrinted);
                printf("%-#8.6g ", phi);
                printf("%-#6.4g ", relRad);
                printf("%-9.5e ", de);
                printf("%9d ", totalBubbleCount);
                printf("%10d ", totalPairCount);
                printf("%6ld ", params.hostData.numStepsInTimeStep);
                printf("%-9ld ", params.hostData.numNeighborsSearched);
                printf("%-9.5e ", minTimestep);
                printf("%-9.5e ", maxTimestep);
                printf("%-9.5e \n",
                       avgTimestep / params.hostData.numStepsInTimeStep);
            }

            ++params.hostData.timesPrinted;
            params.hostData.numStepsInTimeStep = 0;
            params.hostData.energy1 = params.hostData.energy2;
            params.hostData.numNeighborsSearched = 0;
            minTimestep = 9999999.9;
            maxTimestep = -1.0;
            avgTimestep = 0.0;
            resetErrors = true;
        }

        if (isSnapshotTime) {
            saveSnapshot(params);
        }

        if (resetErrors) {
            launchResetArray(params, params.bubbles.count, false, 0.0,
                             params.bubbles.error);
            resetErrors = false;
        }

        ++params.hostData.numStepsInTimeStep;
    }

    if (params.rank == 0) {
        if (totalBubbleCount <= params.hostData.minNumBubbles) {
            printf(
                "Stopping simulation, since the number of bubbles left in the "
                "simulation (%d) is less than or equal to the specified "
                "minimum "
                "(%d)\n",
                totalBubbleCount, params.hostData.minNumBubbles);
        } else if (params.hostData.maxBubbleRadius > minInterval) {
            dvec temp = params.hostConstants.globalInterval;
            printf(
                "Stopping simulation, since the radius of the largest bubble "
                "(%g) is greater than the simulation box (%g, %g, %g)\n",
                params.hostData.maxBubbleRadius, temp.x, temp.y, temp.z);
        } else {
            printf("Stopping simulation for an unknown reason...\n");
        }
    }

    if (params.hostData.snapshotFrequency > 0.0) {
        saveSnapshot(params);
    }

    end(params);
    if (params.rank == 0) {
        printf("Done\n");
    }
}
} // namespace cubble
