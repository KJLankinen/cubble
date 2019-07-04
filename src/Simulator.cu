// -*- C++ -*-

#include "CubWrapper.h"
#include "Env.h"
#include "Kernels.cuh"
#include "Macros.h"
#include "Simulator.cuh"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <cuda_profiler_api.h>
#include <curand.h>
#include <iostream>
#include <nvToolsExt.h>
#include <sstream>
#include <vector>

namespace // anonymous
{
using namespace cubble;

void doBoundaryWrap(KernelSize ks, int sm, cudaStream_t stream, bool wrapX, bool wrapY, bool wrapZ, int numValues,
                    double *x, double *y, double *z, dvec lbb, dvec tfr, int *mulX, int *mulY, int *mulZ, int *mulOldX,
                    int *mulOldY, int *mulOldZ)
{
  if (wrapX && wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y, lbb.y, tfr.y, mulY,
                  mulOldY, z, lbb.z, tfr.z, mulZ, mulOldZ);
  else if (wrapX && wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y, lbb.y, tfr.y, mulY,
                  mulOldY);
  else if (wrapX && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, z, lbb.z, tfr.z, mulZ,
                  mulOldZ);
  else if (wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY, z, lbb.z, tfr.z, mulZ,
                  mulOldZ);
  else if (wrapX)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX);
  else if (wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY);
  else if (wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, z, lbb.z, tfr.z, mulZ, mulOldZ);
}

void doWallVelocity(KernelSize ks, int sm, cudaStream_t stream, bool doX, bool doY, bool doZ, int numValues, int *first,
                    int *second, double *r, double *x, double *y, double *z, double *dxdt, double *dydt, double *dzdt,
                    dvec lbb, dvec tfr, Env properties)
{
  dvec interval = tfr - lbb;
  if (doX && doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX && doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doX && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt);
  }
  else if (doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(), first, second, r,
                  interval.z, lbb.z, !doZ, z, dzdt);
  }
}

void startProfiling(bool start)
{
  if (start)
    cudaProfilerStart();
}

void stopProfiling(bool stop, bool &continueIntegration)
{
  if (stop)
  {
    cudaProfilerStop();
    continueIntegration = false;
  }
}

void reserveMemory(SimulationState &state)
{
  // Reserve pinned memory
  CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&state.pinnedDoubles), 3 * sizeof(double)));
  CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&state.pinnedInts), 1 * sizeof(int)));

  // Calculate the length of 'rows'. Will be divisible by 32, as that's the warp size.
  state.dataStride = state.numBubbles + !!(state.numBubbles % 32) * (32 - state.numBubbles % 32);

  // Doubles
  state.memReqD = sizeof(double) * (uint64_t)state.dataStride * (uint64_t)DDP::NUM_VALUES;
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&state.deviceDoubles), state.memReqD));

  for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
    state.ddps[i] = state.deviceDoubles + i * state.dataStride;

  // Integers
  const uint32_t avgNumNeighbors = 32;
  state.pairStride               = avgNumNeighbors * state.dataStride;

  state.memReqI = sizeof(int) * (uint64_t)state.dataStride *
                  ((uint64_t)DIP::PAIR1 + avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&state.deviceInts), state.memReqI));

  for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
    state.dips[i] = state.deviceInts + i * state.dataStride;

  uint32_t j = 0;
  for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
    state.dips[i] = state.dips[(uint32_t)DIP::PAIR1] + avgNumNeighbors * ++j * state.dataStride;
}

dim3 getGridSize(SimulationState &state, Env &properties)
{
  const int totalNumCells = std::ceil((float)state.numBubbles / properties.getNumBubblesPerCell());
  dvec interval           = properties.getTfr() - properties.getLbb();
  interval                = interval / interval.x;
  float nx                = (float)totalNumCells / interval.y;
  if (NUM_DIM == 3)
    nx = std::cbrt(nx / interval.z);
  else
  {
    nx         = std::sqrt(nx);
    interval.z = 0;
  }

  ivec grid = (nx * interval).floor() + 1;
  assert(grid.x > 0);
  assert(grid.y > 0);
  assert(grid.z > 0);

  return dim3(grid.x, grid.y, grid.z);
}

void updateCellsAndNeighbors(SimulationState &state, Env &properties, CubWrapper &cubWrapper)
{
  dim3 gridSize = getGridSize(state, properties);
  const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

  int *offsets             = state.dips[(uint32_t)DIP::PAIR1];
  int *sizes               = state.dips[(uint32_t)DIP::PAIR1] + state.maxNumCells;
  int *cellIndices         = state.dips[(uint32_t)DIP::TEMP1] + 0 * state.dataStride;
  int *bubbleIndices       = state.dips[(uint32_t)DIP::TEMP1] + 1 * state.dataStride;
  int *sortedCellIndices   = state.dips[(uint32_t)DIP::TEMP1] + 2 * state.dataStride;
  int *sortedBubbleIndices = state.dips[(uint32_t)DIP::TEMP1] + 3 * state.dataStride;

  const size_t resetBytes = sizeof(int) * state.pairStride * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
  CUDA_CALL(cudaMemset(state.dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

  KernelSize kernelSize(128, state.numBubbles);

  KERNEL_LAUNCH(assignBubblesToCells, state.pairKernelSize, 0, 0, state.ddps[(uint32_t)DDP::X],
                state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                properties.getLbb(), properties.getTfr(), cellDim, state.numBubbles);

  cubWrapper.sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices),
                                 sortedCellIndices, const_cast<const int *>(bubbleIndices), sortedBubbleIndices,
                                 state.numBubbles);

  cubWrapper.histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven, cellIndices, sizes,
                                             state.maxNumCells + 1, 0, state.maxNumCells, state.numBubbles);

  cubWrapper.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, state.maxNumCells);

  KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, 0, state.numBubbles, ReorganizeType::COPY_FROM_INDEX,
                sortedBubbleIndices, sortedBubbleIndices, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::XP],
                state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Z],
                state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::RP],
                state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDT],
                state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDT], state.ddps[(uint32_t)DDP::DZDTP],
                state.ddps[(uint32_t)DDP::DRDT], state.ddps[(uint32_t)DDP::DRDTP], state.ddps[(uint32_t)DDP::DXDTO],
                state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::TEMP1],
                state.ddps[(uint32_t)DDP::DZDTO], state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::DRDTO],
                state.ddps[(uint32_t)DDP::TEMP3], state.ddps[(uint32_t)DDP::X0], state.ddps[(uint32_t)DDP::TEMP4],
                state.ddps[(uint32_t)DDP::Y0], state.ddps[(uint32_t)DDP::TEMP5], state.ddps[(uint32_t)DDP::Z0],
                state.ddps[(uint32_t)DDP::TEMP6], state.ddps[(uint32_t)DDP::PATH], state.ddps[(uint32_t)DDP::TEMP7],
                state.ddps[(uint32_t)DDP::DISTANCE], state.ddps[(uint32_t)DDP::TEMP8],
                state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                state.dips[(uint32_t)DIP::WRAP_COUNT_Y], state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(state.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(state.ddps[(uint32_t)DDP::XP]), state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  dvec interval = properties.getTfr() - properties.getLbb();

  kernelSize.block = dim3(128, 1, 1);
  kernelSize.grid  = gridSize;

  CUDA_CALL(cudaMemset(state.np, 0, sizeof(int)));

  const double maxDistance = 1.5 * state.maxBubbleRadius;

  for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
  {
    cudaStream_t stream = (i % 2) ? state.velocityStream : state.gasExchangeStream;
    if (NUM_DIM == 3)
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, state.numBubbles, state.maxNumCells,
                    (int)state.pairStride, maxDistance, offsets, sizes, state.dips[(uint32_t)DIP::TEMP1],
                    state.dips[(uint32_t)DIP::TEMP2], state.ddps[(uint32_t)DDP::R], interval.x, PBC_X == 1,
                    state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], interval.z,
                    PBC_Z == 1, state.ddps[(uint32_t)DDP::Z]);
    else
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, state.numBubbles, state.maxNumCells,
                    (int)state.pairStride, maxDistance, offsets, sizes, state.dips[(uint32_t)DIP::TEMP1],
                    state.dips[(uint32_t)DIP::TEMP2], state.ddps[(uint32_t)DDP::R], interval.x, PBC_X == 1,
                    state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y]);
  }

  CUDA_CALL(cudaMemcpy(static_cast<void *>(state.pinnedInts), state.np, sizeof(int), cudaMemcpyDeviceToHost));
  state.numPairs = state.pinnedInts[0];
  cubWrapper.sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(state.dips[(uint32_t)DIP::TEMP1]),
    state.dips[(uint32_t)DIP::PAIR1], const_cast<const int *>(state.dips[(uint32_t)DIP::TEMP2]),
    state.dips[(uint32_t)DIP::PAIR2], state.numPairs);
}

void deleteSmallBubbles(SimulationState &state, CubWrapper &cubWrapper, int numBubblesAboveMinRad)
{
  NVTX_RANGE_PUSH_A("BubbleRemoval");
  KernelSize kernelSize(128, state.numBubbles);

  CUDA_CALL(cudaMemset(static_cast<void *>(state.dvm), 0, sizeof(double)));
  KERNEL_LAUNCH(calculateRedistributedGasVolume, kernelSize, 0, 0, state.ddps[(uint32_t)DDP::TEMP1],
                state.ddps[(uint32_t)DDP::R], state.dips[(uint32_t)DIP::FLAGS], state.numBubbles);

  cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP1],
                                                      state.dtv, state.numBubbles);

  int *newIdx = state.dips[(uint32_t)DIP::TEMP1];
  cubWrapper.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, state.dips[(uint32_t)DIP::FLAGS], newIdx,
                                state.numBubbles);

  KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, 0, state.numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx,
                state.dips[(uint32_t)DIP::FLAGS], state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::XP],
                state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Z],
                state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::RP],
                state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDT],
                state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDT], state.ddps[(uint32_t)DDP::DZDTP],
                state.ddps[(uint32_t)DDP::DRDT], state.ddps[(uint32_t)DDP::DRDTP], state.ddps[(uint32_t)DDP::DXDTO],
                state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::TEMP1],
                state.ddps[(uint32_t)DDP::DZDTO], state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::DRDTO],
                state.ddps[(uint32_t)DDP::TEMP3], state.ddps[(uint32_t)DDP::X0], state.ddps[(uint32_t)DDP::TEMP4],
                state.ddps[(uint32_t)DDP::Y0], state.ddps[(uint32_t)DDP::TEMP5], state.ddps[(uint32_t)DDP::Z0],
                state.ddps[(uint32_t)DDP::TEMP6], state.ddps[(uint32_t)DDP::PATH], state.ddps[(uint32_t)DDP::TEMP7],
                state.ddps[(uint32_t)DDP::DISTANCE], state.ddps[(uint32_t)DDP::TEMP8],
                state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                state.dips[(uint32_t)DIP::WRAP_COUNT_Y], state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(state.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(state.ddps[(uint32_t)DDP::XP]), state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  state.numBubbles = numBubblesAboveMinRad;
  KERNEL_LAUNCH(addVolume, kernelSize, 0, 0, state.ddps[(uint32_t)DDP::R], state.numBubbles);

  NVTX_RANGE_POP();
}

void setup(SimulationState &state, Env &properties, CubWrapper &cubWrapper)
{
  // First calculate the size of the box and the starting number of bubbles
  dvec relDim        = properties.getBoxRelativeDimensions();
  relDim             = relDim / relDim.x;
  const float d      = 2 * properties.getAvgRad();
  float x            = properties.getNumBubbles() * d * d / relDim.y;
  ivec bubblesPerDim = ivec(0, 0, 0);

  if (NUM_DIM == 3)
  {
    x                = x * d / relDim.z;
    x                = std::cbrt(x);
    relDim           = relDim * x;
    bubblesPerDim    = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), std::ceil(relDim.z / d));
    state.numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
  }
  else
  {
    x                = std::sqrt(x);
    relDim           = relDim * x;
    bubblesPerDim    = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
    state.numBubbles = bubblesPerDim.x * bubblesPerDim.y;
  }

  properties.setTfr(d * bubblesPerDim.asType<double>() + properties.getLbb());
  dvec interval = properties.getTfr() - properties.getLbb();
  properties.setFlowTfr(interval * properties.getFlowTfr() + properties.getLbb());
  properties.setFlowLbb(interval * properties.getFlowLbb() + properties.getLbb());

  // Determine the maximum number of Morton numbers for the simulation box
  dim3 gridDim         = getGridSize(state, properties);
  const int maxGridDim = gridDim.x > gridDim.y ? (gridDim.x > gridDim.z ? gridDim.x : gridDim.z)
                                               : (gridDim.y > gridDim.z ? gridDim.y : gridDim.z);
  int maxNumCells = 1;
  while (maxNumCells < maxGridDim)
    maxNumCells = maxNumCells << 1;

  if (NUM_DIM == 3)
    maxNumCells = maxNumCells * maxNumCells * maxNumCells;
  else
    maxNumCells = maxNumCells * maxNumCells;

  state.maxNumCells = maxNumCells;

  std::cout << "Maximum (theoretical) number of cells: " << state.maxNumCells
            << ", actual grid dimensions: " << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << std::endl;

  // Reserve memory for data
  reserveMemory(state);
  std::cout << "Memory requirement for data:\n\tdouble: " << state.memReqD << " bytes\n\tint: " << state.memReqI
            << " bytes" << std::endl;

  // Get some device global symbol addresses to host pointers.
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dtfa), dTotalFreeArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dtfapr), dTotalFreeAreaPerRadius));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.mbpc), dMaxBubblesPerCell));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dvm), dVolumeMultiplier));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dtv), dTotalVolume));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.np), dNumPairs));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dir), dInvRho));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dta), dTotalArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&state.dasai), dAverageSurfaceAreaIn));

  // Streams
  CUDA_ASSERT(cudaStreamCreate(&state.velocityStream));
  CUDA_ASSERT(cudaStreamCreate(&state.gasExchangeStream));

  printRelevantInfoOfCurrentDevice();

  KernelSize kernelSize(128, state.numBubbles);

  std::cout << "Starting to generate data for bubbles." << std::endl;
  double timeStep        = properties.getTimeStep();
  const int rngSeed      = properties.getRngSeed();
  const double avgRad    = properties.getAvgRad();
  const double stdDevRad = properties.getStdDevRad();
  const dvec tfr         = properties.getTfr();
  const dvec lbb         = properties.getLbb();
  interval               = tfr - lbb;

  curandGenerator_t generator;
  CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
  if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, state.ddps[(uint32_t)DDP::Z], state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, state.ddps[(uint32_t)DDP::X], state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, state.ddps[(uint32_t)DDP::Y], state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, state.ddps[(uint32_t)DDP::RP], state.numBubbles));
  CURAND_CALL(curandGenerateNormalDouble(generator, state.ddps[(uint32_t)DDP::R], state.numBubbles, avgRad, stdDevRad));
  CURAND_CALL(curandDestroyGenerator(generator));

  KERNEL_LAUNCH(assignDataToBubbles, kernelSize, 0, 0, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::Y],
                state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP],
                state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::RP],
                state.dips[(uint32_t)DIP::FLAGS], bubblesPerDim, tfr, lbb, avgRad, properties.getMinRad(),
                state.numBubbles);

  cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::RP],
                                                      state.dasai, state.numBubbles, 0);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(state.ddps[(uint32_t)DDP::RP]),
                            static_cast<void *>(state.ddps[(uint32_t)DDP::R]), sizeof(double) * state.dataStride,
                            cudaMemcpyDeviceToDevice, 0));

  // Delete small bubbles, if any
  const int numBubblesAboveMinRad =
    cubWrapper.reduce<int, int *, int *>(&cub::DeviceReduce::Sum, state.dips[(uint32_t)DIP::FLAGS], state.numBubbles);
  if (numBubblesAboveMinRad < state.numBubbles)
    deleteSmallBubbles(state, cubWrapper, numBubblesAboveMinRad);

  state.maxBubbleRadius = cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Max,
                                                                        state.ddps[(uint32_t)DDP::R], state.numBubbles);

  updateCellsAndNeighbors(state, properties, cubWrapper);

  // Calculate some initial values which are needed
  // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTO],
                state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::DZDTO], state.ddps[(uint32_t)DDP::DRDTO],
                state.ddps[(uint32_t)DDP::DISTANCE], state.ddps[(uint32_t)DDP::PATH]);

  std::cout << "Calculating some initial values as a part of setup." << std::endl;

  if (NUM_DIM == 3)
  {
    KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO],
                  interval.z, lbb.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::X],
                  state.ddps[(uint32_t)DDP::DXDTO], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO],
                  state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::DZDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, state.numBubbles,
                   state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], lbb,
                   tfr, state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                   state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                   state.dips[(uint32_t)DIP::WRAP_COUNT_YP], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTO],
                  state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::DZDTO], state.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO],
                  interval.z, lbb.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::DZDTO]);
  }
  else
  {
    KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::X],
                  state.ddps[(uint32_t)DDP::DXDTO], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, false, state.numBubbles, state.ddps[(uint32_t)DDP::XP],
                   state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], lbb, tfr,
                   state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                   state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                   state.dips[(uint32_t)DIP::WRAP_COUNT_YP], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTO],
                  state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDTO]);
  }
}

void saveSnapshotToFile(SimulationState &state, Env &properties)
{
  std::stringstream ss;
  ss << properties.getSnapshotFilename() << ".csv." << state.numSnapshots;
  std::ofstream file(ss.str().c_str(), std::ios::out);
  if (file.is_open())
  {
    std::vector<double> hostData;
    const size_t numComp = 17;
    hostData.resize(state.dataStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), state.deviceDoubles, sizeof(double) * numComp * state.dataStride,
                         cudaMemcpyDeviceToHost));

    file << "x,y,z,r,vx,vy,vz,path,dist\n";
    for (size_t i = 0; i < (size_t)state.numBubbles; ++i)
    {
      file << hostData[i + 0 * state.dataStride];
      file << ",";
      file << hostData[i + 1 * state.dataStride];
      file << ",";
      file << hostData[i + 2 * state.dataStride];
      file << ",";
      file << hostData[i + 3 * state.dataStride];
      file << ",";
      file << hostData[i + 4 * state.dataStride];
      file << ",";
      file << hostData[i + 5 * state.dataStride];
      file << ",";
      file << hostData[i + 6 * state.dataStride];
      file << ",";
      file << hostData[i + 15 * state.dataStride];
      file << ",";
      file << hostData[i + 16 * state.dataStride];
      file << "\n";
    }

    ++state.numSnapshots;
  }
}

double stabilize(SimulationState &state, Env &properties, CubWrapper &cubWrapper)
{
  // This function integrates only the positions of the bubbles.
  // Gas exchange is not used. This is used for equilibrating the foam.

  KernelSize kernelSize(128, state.numBubbles);

  const dvec tfr      = properties.getTfr();
  const dvec lbb      = properties.getLbb();
  const dvec interval = tfr - lbb;
  double elapsedTime  = 0.0;
  double timeStep     = properties.getTimeStep();
  double error        = 100000;

  // Energy before stabilization
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles, state.dips[(uint32_t)DIP::PAIR1],
                  state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4],
                  interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                  state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles, state.dips[(uint32_t)DIP::PAIR1],
                  state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4],
                  interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                  state.ddps[(uint32_t)DDP::Y]);

  cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP4],
                                                      state.dtfapr, state.numBubbles);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&state.pinnedDoubles[1]), static_cast<void *>(state.dtfapr),
                            sizeof(double), cudaMemcpyDeviceToHost, 0));

  for (int i = 0; i < properties.getNumStepsToRelax(); ++i)
  {
    do
    {
      if (NUM_DIM == 3)
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTP],
                      state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDTP],
                      state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::TEMP1],
                      state.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::XP],
                      state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTO],
                      state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT],
                      state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::Z],
                      state.ddps[(uint32_t)DDP::DZDT], state.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                      interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::DXDTP],
                      interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::DYDTP],
                      interval.z, lbb.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::DZDTP]);

        doWallVelocity(state.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, state.numBubbles,
                       state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2],
                       state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP],
                       state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::DXDTP],
                       state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::ERROR],
                      state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT],
                      state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y],
                      state.ddps[(uint32_t)DDP::DYDT], state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::ZP],
                      state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::DZDT], state.ddps[(uint32_t)DDP::DZDTP]);
      }
      else // Two dimensional case
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTP],
                      state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::ERROR],
                      state.ddps[(uint32_t)DDP::TEMP1], state.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::XP],
                      state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTO],
                      state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT],
                      state.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                      interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::DXDTP],
                      interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::DYDTP]);

        doWallVelocity(state.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, state.numBubbles,
                       state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2],
                       state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP],
                       state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::DXDTP],
                       state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::ERROR],
                      state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT],
                      state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y],
                      state.ddps[(uint32_t)DDP::DYDT], state.ddps[(uint32_t)DDP::DYDTP]);
      }

      doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, state.numBubbles,
                     state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_YP], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Error
      error = cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Max, state.ddps[(uint32_t)DDP::ERROR],
                                                            state.numBubbles);

      if (error < properties.getErrorTolerance() && timeStep < 0.1)
        timeStep *= 1.9;
      else if (error > properties.getErrorTolerance())
        timeStep *= 0.5;

    } while (error > properties.getErrorTolerance());

    // Update the current values with the calculated predictions
    const size_t numBytesToCopy = 3 * sizeof(double) * state.dataStride;
    CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::DXDTO], state.ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::XP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));

    properties.setTimeStep(timeStep);
    elapsedTime += timeStep;

    if (i % 5000 == 0)
      updateCellsAndNeighbors(state, properties, cubWrapper);
  }

  // Energy after stabilization
  state.energy1 = state.pinnedDoubles[1];

  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles, state.dips[(uint32_t)DIP::PAIR1],
                  state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4],
                  interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                  state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles, state.dips[(uint32_t)DIP::PAIR1],
                  state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4],
                  interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                  state.ddps[(uint32_t)DDP::Y]);

  state.energy2 = cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                state.ddps[(uint32_t)DDP::TEMP4], state.numBubbles);

  return elapsedTime;
}

bool integrate(SimulationState &state, Env &properties, CubWrapper &cubWrapper)
{
  NVTX_RANGE_PUSH_A("Integration function");
  KernelSize kernelSize(128, state.numBubbles);

  const dvec tfr        = properties.getTfr();
  const dvec lbb        = properties.getLbb();
  const dvec interval   = tfr - lbb;
  double timeStep       = properties.getTimeStep();
  double error          = 100000;
  uint32_t numLoopsDone = 0;

  do
  {
    NVTX_RANGE_PUSH_A("Integration step");

    if (NUM_DIM == 3)
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTP],
                    state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DZDTP],
                    state.ddps[(uint32_t)DDP::DRDTP], state.ddps[(uint32_t)DDP::ERROR],
                    state.ddps[(uint32_t)DDP::TEMP1], state.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::XP],
                    state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTO],
                    state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT],
                    state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::Z],
                    state.ddps[(uint32_t)DDP::DZDT], state.ddps[(uint32_t)DDP::DZDTO], state.ddps[(uint32_t)DDP::RP],
                    state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::DRDT], state.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, state.velocityStream, properties.getFZeroPerMuZero(),
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                    interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::DXDTP],
                    interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::DYDTP],
                    interval.z, lbb.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::DZDTP]);
      // Wall velocity
      doWallVelocity(state.pairKernelSize, 0, state.velocityStream, PBC_X == 0, PBC_Y == 0, PBC_Z == 0,
                     state.numBubbles, state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2],
                     state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP],
                     state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDTP],
                     state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(state.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * state.pairStride));
        int *numNeighbors = state.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, state.pairKernelSize, 0, state.velocityStream,
                      state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      state.ddps[(uint32_t)DDP::TEMP1], state.ddps[(uint32_t)DDP::DXDTO],
                      state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::DYDTO],
                      state.ddps[(uint32_t)DDP::TEMP3], state.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, state.pairKernelSize, 0, state.velocityStream, state.numBubbles, numNeighbors,
                      state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDTP],
                      state.ddps[(uint32_t)DDP::DZDTP], state.ddps[(uint32_t)DDP::TEMP1],
                      state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::TEMP3], state.ddps[(uint32_t)DDP::XP],
                      state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], properties.getFlowVel(),
                      properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, state.velocityStream, state.numBubbles, timeStep,
                    state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X],
                    state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::YP],
                    state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT], state.ddps[(uint32_t)DDP::DYDTP],
                    state.ddps[(uint32_t)DDP::ZP], state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::DZDT],
                    state.ddps[(uint32_t)DDP::DZDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, state.velocityStream, state.numBubbles,
                    state.ddps[(uint32_t)DDP::TEMP4], state.ddps[(uint32_t)DDP::PATH],
                    state.ddps[(uint32_t)DDP::DISTANCE], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X],
                    state.ddps[(uint32_t)DDP::X0], state.dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x,
                    state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::Y0],
                    state.dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y, state.ddps[(uint32_t)DDP::ZP],
                    state.ddps[(uint32_t)DDP::Z], state.ddps[(uint32_t)DDP::Z0],
                    state.dips[(uint32_t)DIP::WRAP_COUNT_ZP], interval.z);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, state.velocityStream, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, state.numBubbles,
                     state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_YP], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, state.pairKernelSize, 0, state.gasExchangeStream, state.numBubbles,
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                    state.ddps[(uint32_t)DDP::DRDTP], state.ddps[(uint32_t)DDP::TEMP1], interval.x, PBC_X == 1,
                    state.ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP], interval.z,
                    PBC_Z == 1, state.ddps[(uint32_t)DDP::ZP]);
    }
    else // Two dimensions
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::DXDTP],
                    state.ddps[(uint32_t)DDP::DYDTP], state.ddps[(uint32_t)DDP::DRDTP],
                    state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::TEMP1],
                    state.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, state.numBubbles, timeStep, state.ddps[(uint32_t)DDP::XP],
                    state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTO],
                    state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT],
                    state.ddps[(uint32_t)DDP::DYDTO], state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::R],
                    state.ddps[(uint32_t)DDP::DRDT], state.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, state.pairKernelSize, 0, state.velocityStream, properties.getFZeroPerMuZero(),
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                    interval.x, lbb.x, PBC_X == 1, state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::DXDTP],
                    interval.y, lbb.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::DYDTP]);
      // Wall velocity
      doWallVelocity(state.pairKernelSize, 0, state.velocityStream, PBC_X == 0, PBC_Y == 0, false, state.numBubbles,
                     state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                     state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP],
                     state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDTP],
                     state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(state.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * state.pairStride));
        int *numNeighbors = state.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, state.pairKernelSize, 0, state.velocityStream,
                      state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      state.ddps[(uint32_t)DDP::TEMP1], state.ddps[(uint32_t)DDP::DXDTO],
                      state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, state.pairKernelSize, 0, state.velocityStream, state.numBubbles, numNeighbors,
                      state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::DYDTP],
                      state.ddps[(uint32_t)DDP::DZDTP], state.ddps[(uint32_t)DDP::TEMP1],
                      state.ddps[(uint32_t)DDP::TEMP2], state.ddps[(uint32_t)DDP::TEMP3], state.ddps[(uint32_t)DDP::XP],
                      state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], properties.getFlowVel(),
                      properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, state.velocityStream, state.numBubbles, timeStep,
                    state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X],
                    state.ddps[(uint32_t)DDP::DXDT], state.ddps[(uint32_t)DDP::DXDTP], state.ddps[(uint32_t)DDP::YP],
                    state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::DYDT], state.ddps[(uint32_t)DDP::DYDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, state.velocityStream, state.numBubbles,
                    state.ddps[(uint32_t)DDP::TEMP4], state.ddps[(uint32_t)DDP::PATH],
                    state.ddps[(uint32_t)DDP::DISTANCE], state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::X],
                    state.ddps[(uint32_t)DDP::X0], state.dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x,
                    state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::Y], state.ddps[(uint32_t)DDP::Y0],
                    state.dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, state.velocityStream, PBC_X == 1, PBC_Y == 1, false, state.numBubbles,
                     state.ddps[(uint32_t)DDP::XP], state.ddps[(uint32_t)DDP::YP], state.ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, state.dips[(uint32_t)DIP::WRAP_COUNT_X], state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_Z], state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     state.dips[(uint32_t)DIP::WRAP_COUNT_YP], state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, state.pairKernelSize, 0, state.gasExchangeStream, state.numBubbles,
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::RP],
                    state.ddps[(uint32_t)DDP::DRDTP], state.ddps[(uint32_t)DDP::TEMP1], interval.x, PBC_X == 1,
                    state.ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::YP]);
    }

    // Free area
    KERNEL_LAUNCH(freeAreaKernel, kernelSize, 0, state.gasExchangeStream, state.numBubbles,
                  state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::TEMP1], state.ddps[(uint32_t)DDP::TEMP2],
                  state.ddps[(uint32_t)DDP::TEMP3]);

    cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP1],
                                                        state.dtfa, state.numBubbles, state.gasExchangeStream);
    cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP2],
                                                        state.dtfapr, state.numBubbles, state.gasExchangeStream);
    cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP3],
                                                        state.dta, state.numBubbles, state.gasExchangeStream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, kernelSize, 0, state.gasExchangeStream, state.ddps[(uint32_t)DDP::DRDTP],
                  state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::TEMP1], state.numBubbles,
                  properties.getKappa(), properties.getKParameter());

    // Radius correct
    KERNEL_LAUNCH(correctKernel, kernelSize, 0, state.gasExchangeStream, state.numBubbles, timeStep,
                  state.ddps[(uint32_t)DDP::ERROR], state.ddps[(uint32_t)DDP::RP], state.ddps[(uint32_t)DDP::R],
                  state.ddps[(uint32_t)DDP::DRDT], state.ddps[(uint32_t)DDP::DRDTP]);

    // Calculate how many bubbles are below the minimum size.
    // Also take note of maximum radius.
    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, kernelSize, 0, state.gasExchangeStream, state.numBubbles,
                  state.dips[(uint32_t)DIP::FLAGS], state.ddps[(uint32_t)DDP::RP], properties.getMinRad());

    cubWrapper.reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, state.dips[(uint32_t)DIP::FLAGS],
                                               static_cast<int *>(state.mbpc), state.numBubbles,
                                               state.gasExchangeStream);
    cubWrapper.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, state.ddps[(uint32_t)DDP::RP],
                                                        static_cast<double *>(state.dtfa), state.numBubbles,
                                                        state.gasExchangeStream);

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(state.pinnedInts), state.mbpc, sizeof(int), cudaMemcpyDeviceToHost,
                              state.gasExchangeStream));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(state.pinnedDoubles), state.dtfa, sizeof(double),
                              cudaMemcpyDeviceToHost, state.gasExchangeStream));

    // Error
    error = cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Max, state.ddps[(uint32_t)DDP::ERROR],
                                                          state.numBubbles);

    if (error < properties.getErrorTolerance() && timeStep < 0.1)
      timeStep *= 1.9;
    else if (error > properties.getErrorTolerance())
      timeStep *= 0.5;

    ++numLoopsDone;

    NVTX_RANGE_POP();
  } while (error > properties.getErrorTolerance());

  // Update values
  const size_t numBytesToCopy = 4 * sizeof(double) * state.dataStride;

  CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::DXDTO], state.ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::XP], 2 * numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(state.ddps[(uint32_t)DDP::PATH], state.ddps[(uint32_t)DDP::TEMP4],
                            sizeof(double) * state.dataStride, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(state.dips[(uint32_t)DIP::WRAP_COUNT_XP], state.dips[(uint32_t)DIP::WRAP_COUNT_X],
                            state.dataStride * 3 * sizeof(int), cudaMemcpyDeviceToDevice));

  ++state.integrationStep;
  properties.setTimeStep(timeStep);
  state.simulationTime += timeStep;

  state.maxBubbleRadius = state.pinnedDoubles[0];

  // Delete & reorder
  const int numBubblesAboveMinRad = state.pinnedInts[0];
  const bool shouldDeleteBubbles  = numBubblesAboveMinRad < state.numBubbles;

  if (shouldDeleteBubbles)
    deleteSmallBubbles(state, cubWrapper, numBubblesAboveMinRad);

  if (shouldDeleteBubbles || state.integrationStep % 5000 == 0)
    updateCellsAndNeighbors(state, properties, cubWrapper);

  bool continueSimulation = state.numBubbles > properties.getMinNumBubbles();
  continueSimulation &= (NUM_DIM == 3) ? state.maxBubbleRadius < 0.5 * (tfr - lbb).getMinComponent() : true;

  NVTX_RANGE_POP();

  return continueSimulation;
}

void transformPositions(SimulationState &state, Env &properties, bool normalize)
{
  KERNEL_LAUNCH(transformPositionsKernel, state.pairKernelSize, 0, 0, normalize, state.numBubbles, properties.getLbb(),
                properties.getTfr(), state.ddps[(uint32_t)DDP::X], state.ddps[(uint32_t)DDP::Y],
                state.ddps[(uint32_t)DDP::Z]);
}

double calculateVolumeOfBubbles(SimulationState &state, CubWrapper &cubWrapper)
{
  KernelSize kernelSize(128, state.numBubbles);

  KERNEL_LAUNCH(calculateVolumes, kernelSize, 0, 0, state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP1],
                state.numBubbles);

  return cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP1],
                                                       state.numBubbles);
}

void deinit(SimulationState &state, Env &properties)
{
  saveSnapshotToFile(state, properties);
  properties.writeParameters();

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(static_cast<void *>(state.deviceDoubles)));
  CUDA_CALL(cudaFree(static_cast<void *>(state.deviceInts)));
  CUDA_CALL(cudaFreeHost(static_cast<void *>(state.pinnedInts)));
  CUDA_CALL(cudaFreeHost(static_cast<void *>(state.pinnedDoubles)));

  CUDA_CALL(cudaStreamDestroy(state.velocityStream));
  CUDA_CALL(cudaStreamDestroy(state.gasExchangeStream));
}
} // namespace

namespace cubble
{
void run(const char *inputFileName, const char *outputFileName)
{
  CubWrapper cubWrapper;
  SimulationState state;
  Env properties = Env(inputFileName, outputFileName);
  properties.readParameters();

  std::cout << "\n=====\nSetup\n=====" << std::endl;
  {
    setup(state, properties, cubWrapper);
    saveSnapshotToFile(state, properties);

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;
    stabilize(state, properties, cubWrapper);
    saveSnapshotToFile(state, properties);

    const double phiTarget = properties.getPhiTarget();
    double bubbleVolume    = calculateVolumeOfBubbles(state, cubWrapper);
    double phi             = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;
    transformPositions(state, properties, true);
    dvec relativeSize = properties.getBoxRelativeDimensions();
    relativeSize.z    = (NUM_DIM == 2) ? 1 : relativeSize.z;
    double t =
      calculateVolumeOfBubbles(state, cubWrapper) / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z);
    t = (NUM_DIM == 3) ? std::cbrt(t) : std::sqrt(t);
    properties.setTfr(dvec(t, t, t) * relativeSize);
    transformPositions(state, properties, false);

    phi = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    saveSnapshotToFile(state, properties);
  }

  std::cout << "\n=============\nStabilization\n=============" << std::endl;
  {
    int numSteps       = 0;
    const int failsafe = 500;

    std::cout << "#steps\tdE/t\te1\te2" << std::endl;
    while (true)
    {
      double time        = stabilize(state, properties, cubWrapper);
      double deltaEnergy = std::abs(state.energy2 - state.energy1) / time;
      deltaEnergy *= 0.5 * properties.getSigmaZero();

      if (deltaEnergy < properties.getMaxDeltaEnergy())
      {
        std::cout << "Final delta energy " << deltaEnergy << " after "
                  << (numSteps + 1) * properties.getNumStepsToRelax() << " steps."
                  << " Energy before: " << state.energy1 << ", energy after: " << state.energy2
                  << ", time: " << time * properties.getTimeScalingFactor() << std::endl;
        break;
      }
      else if (numSteps > failsafe)
      {
        std::cout << "Over " << failsafe * properties.getNumStepsToRelax()
                  << " steps taken and required delta energy not reached."
                  << " Check parameters." << std::endl;
        break;
      }
      else
      {
        std::cout << (numSteps + 1) * properties.getNumStepsToRelax() << "\t" << deltaEnergy << "\t" << state.energy1
                  << "\t" << state.energy2 << std::endl;
      }

      ++numSteps;
    }

    saveSnapshotToFile(state, properties);
  }

  std::cout << "\n==========\nSimulation\n==========" << std::endl;
  std::stringstream dataStream;
  {
    // Set starting positions and reset wrap counts to 0
    const size_t numBytesToCopy = 3 * sizeof(double) * state.dataStride;
    CUDA_CALL(cudaMemcpy(state.ddps[(uint32_t)DDP::X0], state.ddps[(uint32_t)DDP::X], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemset(state.dips[(uint32_t)DIP::WRAP_COUNT_X], 0, 6 * state.dataStride * sizeof(int)));

    state.simulationTime = 0;
    int timesPrinted     = 1;
    uint32_t numSteps    = 0;
    const dvec interval  = properties.getTfr() - properties.getLbb();

    KernelSize kernelSize(128, state.numBubbles);

    // Calculate the energy at simulation start
    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::TEMP4]);

    if (NUM_DIM == 3)
    {
      KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles,
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                    state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y,
                    PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1, state.ddps[(uint32_t)DDP::Z]);
    }
    else
    {
      KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles,
                    state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2], state.ddps[(uint32_t)DDP::R],
                    state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, state.ddps[(uint32_t)DDP::X], interval.y,
                    PBC_Y == 1, state.ddps[(uint32_t)DDP::Y]);
    }

    state.energy1 = cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                  state.ddps[(uint32_t)DDP::TEMP4], state.numBubbles);

    // Start the simulation proper
    bool continueIntegration = true;
    int numTotalSteps        = 0;
    std::cout << "T\tphi\tR\t#b\tdE\t\t#steps\t#pairs" << std::endl;
    while (continueIntegration)
    {
      continueIntegration = integrate(state, properties, cubWrapper);
      CUDA_PROFILER_START(numTotalSteps == 2000);
      CUDA_PROFILER_STOP(numTotalSteps == 2200, continueIntegration);

      // The if clause contains many slow operations, but it's only done
      // very few times relative to the entire run time, so it should not
      // have a huge cost. Roughly 6e4-1e5 integration steps are taken for each
      // time step
      // and the if clause is executed once per time step.
      const double scaledTime = state.simulationTime * properties.getTimeScalingFactor();
      if ((int)scaledTime >= timesPrinted)
      {
        kernelSize = KernelSize(128, state.numBubbles);
        // Calculate total energy
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, state.numBubbles, state.ddps[(uint32_t)DDP::TEMP4]);

        if (NUM_DIM == 3)
          KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles,
                        state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2],
                        state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                        state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y], interval.z,
                        PBC_Z == 1, state.ddps[(uint32_t)DDP::Z]);
        else
          KERNEL_LAUNCH(potentialEnergyKernel, state.pairKernelSize, 0, 0, state.numBubbles,
                        state.dips[(uint32_t)DIP::PAIR1], state.dips[(uint32_t)DIP::PAIR2],
                        state.ddps[(uint32_t)DDP::R], state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                        state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, state.ddps[(uint32_t)DDP::Y]);

        state.energy2 = cubWrapper.reduce<double, double *, double *>(
          &cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::TEMP4], state.numBubbles);
        const double dE = (state.energy2 - state.energy1) / state.energy2;

        // Add values to data stream
        const double averageRadius = cubWrapper.reduce<double, double *, double *>(
                                       &cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::R], state.numBubbles) /
                                     state.numBubbles;
        const double averagePath = cubWrapper.reduce<double, double *, double *>(
                                     &cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::PATH], state.numBubbles) /
                                   state.numBubbles;
        const double averageDistance =
          cubWrapper.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, state.ddps[(uint32_t)DDP::DISTANCE],
                                                        state.numBubbles) /
          state.numBubbles;
        const double relativeRadius = averageRadius / properties.getAvgRad();
        dataStream << scaledTime << " " << relativeRadius << " " << state.maxBubbleRadius / properties.getAvgRad()
                   << " " << state.numBubbles << " " << averagePath << " " << averageDistance << " " << dE << "\n";

        // Print some values
        std::cout << (int)scaledTime << "\t"
                  << calculateVolumeOfBubbles(state, cubWrapper) / properties.getSimulationBoxVolume() << "\t"
                  << relativeRadius << "\t" << state.numBubbles << "\t" << dE << "\t" << numSteps << "\t"
                  << state.numPairs << std::endl;

        // Only write snapshots when t* is a power of 2.
        if ((timesPrinted & (timesPrinted - 1)) == 0)
          saveSnapshotToFile(state, properties);

        ++timesPrinted;
        numSteps      = 0;
        state.energy1 = state.energy2;
      }

      ++numSteps;
      ++numTotalSteps;
    }
  }

  std::ofstream file(properties.getDataFilename());
  file << dataStream.str() << std::endl;

  deinit(state, properties);
}
} // namespace cubble
