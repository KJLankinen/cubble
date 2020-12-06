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
#include "Util.h"
#include "Vec.h"

namespace cubble {
double totalEnergy(Params &params);
double totalVolume(Params &params);
void removeBubbles(Params &params, int numToBeDeleted);
void launchInitGlobals(Params &params);
void launchPreIntegrate(Params &params, IntegrationParams &ip);
void launchGatherOutgoingBubbles(Params &params);
void launchPostIntegrate(Params &params, IntegrationParams &ip);
void launchPairwiseInteraction(Params &params, IntegrationParams &ip,
                               uint32_t dynSharedMemBytes);
void launchExternalPairwiseInteraction(Params &params, IntegrationParams &ip);
void launchPotentialEnergy(Params &params);
double getSum(Params &params, double *p);
void launchFindSurfaceCells(Params &params, int numCells, int *surfaceCells,
                            int *cellSizes, int *surfaceCellSizes,
                            int *bubbleCountPerArea, ivec cellDim);
void cubExclusiveSum(void *tempMem, uint64_t maxCubMem, int *src, int *dst,
                     int n, cudaStream_t stream, bool debug);
void cubInclusiveSum(void *tempMem, uint64_t maxCubMem, int *src, int *dst,
                     int n, cudaStream_t stream, bool debug);
void cubSum(void *tempMem, uint64_t maxCubMem, double *src, void *dst, int n,
            cudaStream_t stream, bool debug);
void cubMax(void *tempMem, uint64_t maxCubMem, double *src, void *dst, int n,
            cudaStream_t stream, bool debug);
void launchGatherSurfaceBubbles(Params &params, int nSurfaceCells,
                                int *surfaceCells, int *surfaceCellOffsets,
                                int *cellSizes, int *cellOffsets, double *x,
                                double *y, double *z, double *r, int *idx,
                                int *sizes);
void launchNeighborSearch(Params &params, int numCells, bool internalSearch,
                          int numNeighborCells, ivec cellDim, int *offsets,
                          int *sizes, int *histogram, int *pairI, int *pairJ,
                          SurfaceData &surfaceData, int *surfaceCells,
                          int *procNum);
void launchSortExternalPairs(Params &params, int *pair1, int *pair2,
                             int *offsets, int *procNum,
                             ExternalBubbles::Data &data);
void launchWrapOverPeriodicBoundaries(Params &params, int *indices,
                                      int *procNums, int *procSizes);
void launchGatherAndDeleteMovedBubbles(Params &params, int numToMove,
                                       int bytesPerBubble, int *procSizes,
                                       int *procGlobalOffsets,
                                       int *procLocalOffsets, int *movedIndices,
                                       int *procNums, char *data);
void launchDistributeReceivedBubbles(Params &params, int numReceivedBubbles,
                                     int bytesPerBubble, int *procSizes,
                                     int *procGlobalOffsets,
                                     int *procLocalOffsets, int *procNums,
                                     char *dst);
void launchCellByPosition(Params &params, int *cellIndices, int *cellSizes,
                          ivec cellDim);
void launchIndexByCell(Params &params, int *cellIndices, int *cellOffsets,
                       int *bubbleIndices, int n);
void launchReorganizeByIndex(Params &params, const int *bubbleIndices);
void launchSortPairs(Params &params);
void launchCountNumNeighbors(Params &params);
void launchAssignDataToBubbles(Params &params, ivec bubblesPerDim);
void launchEuler(Params &params);
void launchTransformPositions(Params &params, bool normalize);
void launchResetArray(Params &params, int n, bool resetGlobals, double val,
                      double *p);
void setNumPairsToZero();
void setNumToBeDeletedToZero();
void setIncomingExternalPairsToZero();
void getIncomingExternalPairs(void *data, uint32_t bytes);
void setOutgoingExternalPairs(void *data, uint32_t bytes);
void setAreaToProcessorMap(void *data, uint32_t bytes);
void setConstants(void *data, uint32_t bytes);
void getMaxRadius(void *data, uint32_t bytes);
void getNumToBeDeleted(void *data, uint32_t bytes, bool async);
void getNumPairs(void *data, uint32_t bytes);
void getAreaTotals(std::array<double, 4> &arr);
void setAreaTotals(std::array<double, 4> &arr);
void getTotalVolumeNew(double *data);
void setTotalVolumeNew(double *data);
} // namespace cubble
