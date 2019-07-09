#include "CubWrapper.h"
#include "Kernels.cuh"
#include "Macros.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include "nlohmann/json.hpp"
#include <array>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <nvToolsExt.h>
#include <sstream>
#include <string>
#include <vector>

namespace cubble
{
// Device double pointer names
enum class DDP
{
  X,
  Y,
  Z,
  R,

  DXDT,
  DYDT,
  DZDT,
  DRDT,

  DXDTO,
  DYDTO,
  DZDTO,
  DRDTO,

  X0,
  Y0,
  Z0,

  PATH,
  DISTANCE,

  XP,
  YP,
  ZP,
  RP,

  DXDTP,
  DYDTP,
  DZDTP,
  DRDTP,

  ERROR,

  TEMP1,
  TEMP2,
  TEMP3,
  TEMP4,
  TEMP5,
  TEMP6,
  TEMP7,
  TEMP8,

  NUM_VALUES
};

// Device int pointer names
enum class DIP
{
  FLAGS,

  WRAP_COUNT_X,
  WRAP_COUNT_Y,
  WRAP_COUNT_Z,

  WRAP_COUNT_XP,
  WRAP_COUNT_YP,
  WRAP_COUNT_ZP,

  PAIR1,
  PAIR2,

  TEMP1,
  TEMP2,

  NUM_VALUES
};

struct SimulationState
{
  int numBubbles        = 0;
  int maxNumCells       = 0;
  int numPairs          = 0;
  uint32_t numSnapshots = 0;
  uint32_t timesPrinted = 0;
  uint32_t dataStride   = 0;
  uint32_t pairStride   = 0;

  uint64_t memReqD             = 0;
  uint64_t memReqI             = 0;
  uint64_t numIntegrationSteps = 0;
  uint64_t numStepsInTimeStep  = 0;
  double simulationTime        = 0.0;
  double energy1               = 0.0;
  double energy2               = 0.0;
  double maxBubbleRadius       = 0.0;
  double timeStep              = 0.0;
  double averageSurfaceAreaIn  = 0.0;

  dvec lbb      = dvec(0.0, 0.0, 0.0);
  dvec tfr      = dvec(0.0, 0.0, 0.0);
  dvec interval = dvec(0.0, 0.0, 0.0);
};

struct SimulationInputs
{
  int numBubblesPerCell = 0;
  int rngSeed           = 0;
  int numStepsToRelax   = 0;
  int numBubblesIn      = 0;
  int minNumBubbles     = 0;

  double avgRad            = 0.0;
  double stdDevRad         = 0.0;
  double minRad            = 0.0;
  double phiTarget         = 0.0;
  double muZero            = 0.0;
  double sigmaZero         = 0.0;
  double fZeroPerMuZero    = 0.0;
  double errorTolerance    = 0.0;
  double maxDeltaEnergy    = 0.0;
  double kParameter        = 0.0;
  double kappa             = 0.0;
  double timeScalingFactor = 0.0;
  double timeStepIn        = 0.0;

  dvec boxRelDim = dvec(0.0, 0.0, 0.0);
  dvec flowLbb   = dvec(0.0, 0.0, 0.0);
  dvec flowTfr   = dvec(0.0, 0.0, 0.0);
  dvec flowVel   = dvec(0.0, 0.0, 0.0);

  std::string snapshotFilename = "";
  std::string dataFilename     = "";
};

struct Params
{
  SimulationState state;
  SimulationInputs inputs;
  CubWrapper cw;

  cudaStream_t velocityStream;
  cudaStream_t gasStream;

  KernelSize pairKernelSize = KernelSize(dim3(256, 1, 1), dim3(128, 1, 1));
  KernelSize defaultKernelSize;

  // Device memory & arrays of pointers to those memory chunks.
  int *deviceIntMemory       = nullptr;
  double *deviceDoubleMemory = nullptr;
  std::array<double *, (uint64_t)DDP::NUM_VALUES> ddps;
  std::array<int *, (uint64_t)DIP::NUM_VALUES> dips;
};

} // namespace cubble

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
                    dvec lbb, dvec tfr, Params &params)
{
  dvec interval = tfr - lbb;
  if (doX && doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX && doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doX && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt);
  }
  else if (doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
                  interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.inputs.fZeroPerMuZero, first, second, r,
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

dim3 getGridSize(Params &params)
{
  const int totalNumCells = std::ceil((float)params.state.numBubbles / params.inputs.numBubblesPerCell);
  dvec relativeInterval   = params.state.interval / params.state.interval.x;
  float nx                = (float)totalNumCells / relativeInterval.y;
  if (NUM_DIM == 3)
    nx = std::cbrt(nx / relativeInterval.z);
  else
  {
    nx                 = std::sqrt(nx);
    relativeInterval.z = 0;
  }

  ivec grid = (nx * relativeInterval).floor() + 1;
  assert(grid.x > 0);
  assert(grid.y > 0);
  assert(grid.z > 0);

  return dim3(grid.x, grid.y, grid.z);
}

void updateCellsAndNeighbors(Params &params)
{
  dim3 gridSize = getGridSize(params);
  const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

  int *offsets             = params.dips[(uint32_t)DIP::PAIR1];
  int *sizes               = params.dips[(uint32_t)DIP::PAIR1] + params.state.maxNumCells;
  int *cellIndices         = params.dips[(uint32_t)DIP::TEMP1] + 0 * params.state.dataStride;
  int *bubbleIndices       = params.dips[(uint32_t)DIP::TEMP1] + 1 * params.state.dataStride;
  int *sortedCellIndices   = params.dips[(uint32_t)DIP::TEMP1] + 2 * params.state.dataStride;
  int *sortedBubbleIndices = params.dips[(uint32_t)DIP::TEMP1] + 3 * params.state.dataStride;

  const size_t resetBytes = sizeof(int) * params.state.pairStride * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
  CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

  KERNEL_LAUNCH(assignBubblesToCells, params.pairKernelSize, 0, 0, params.ddps[(uint32_t)DDP::X],
                params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                params.state.lbb, params.state.tfr, cellDim, params.state.numBubbles);

  params.cw.sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices),
                                sortedCellIndices, const_cast<const int *>(bubbleIndices), sortedBubbleIndices,
                                params.state.numBubbles);

  params.cw.histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven, cellIndices, sizes,
                                            params.state.maxNumCells + 1, 0, params.state.maxNumCells,
                                            params.state.numBubbles);

  params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, params.state.maxNumCells);

  KERNEL_LAUNCH(reorganizeKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles,
                ReorganizeType::COPY_FROM_INDEX, sortedBubbleIndices, sortedBubbleIndices,
                params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::DXDT],
                params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::DRDT],
                params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::ERROR],
                params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::DZDTO],
                params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::TEMP3],
                params.ddps[(uint32_t)DDP::X0], params.ddps[(uint32_t)DDP::TEMP4], params.ddps[(uint32_t)DDP::Y0],
                params.ddps[(uint32_t)DDP::TEMP5], params.ddps[(uint32_t)DDP::Z0], params.ddps[(uint32_t)DDP::TEMP6],
                params.ddps[(uint32_t)DDP::PATH], params.ddps[(uint32_t)DDP::TEMP7],
                params.ddps[(uint32_t)DDP::DISTANCE], params.ddps[(uint32_t)DDP::TEMP8],
                params.dips[(uint32_t)DIP::WRAP_COUNT_X], params.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                params.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                params.dips[(uint32_t)DIP::WRAP_COUNT_Z], params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(params.ddps[(uint32_t)DDP::XP]), params.state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  KernelSize kernelSizeNeighbor = KernelSize(gridSize, dim3(128, 1, 1));
  const double maxDistance      = 1.5 * params.state.maxBubbleRadius;

  int *dnp = nullptr;
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dnp), dNumPairs));
  CUDA_CALL(cudaMemset(dnp, 0, sizeof(int)));

  for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
  {
    cudaStream_t stream = (i % 2) ? params.velocityStream : params.gasStream;
    if (NUM_DIM == 3)
      KERNEL_LAUNCH(neighborSearch, kernelSizeNeighbor, 0, stream, i, params.state.numBubbles, params.state.maxNumCells,
                    (int)params.state.pairStride, maxDistance, offsets, sizes, params.dips[(uint32_t)DIP::TEMP1],
                    params.dips[(uint32_t)DIP::TEMP2], params.ddps[(uint32_t)DDP::R], params.state.interval.x,
                    PBC_X == 1, params.ddps[(uint32_t)DDP::X], params.state.interval.y, PBC_Y == 1,
                    params.ddps[(uint32_t)DDP::Y], params.state.interval.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::Z]);
    else
      KERNEL_LAUNCH(neighborSearch, kernelSizeNeighbor, 0, stream, i, params.state.numBubbles, params.state.maxNumCells,
                    (int)params.state.pairStride, maxDistance, offsets, sizes, params.dips[(uint32_t)DIP::TEMP1],
                    params.dips[(uint32_t)DIP::TEMP2], params.ddps[(uint32_t)DDP::R], params.state.interval.x,
                    PBC_X == 1, params.ddps[(uint32_t)DDP::X], params.state.interval.y, PBC_Y == 1,
                    params.ddps[(uint32_t)DDP::Y]);
  }

  CUDA_CALL(cudaMemcpy(static_cast<void *>(&params.state.numPairs), dnp, sizeof(int), cudaMemcpyDeviceToHost));
  params.cw.sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(params.dips[(uint32_t)DIP::TEMP1]),
    params.dips[(uint32_t)DIP::PAIR1], const_cast<const int *>(params.dips[(uint32_t)DIP::TEMP2]),
    params.dips[(uint32_t)DIP::PAIR2], params.state.numPairs);
}

void deleteSmallBubbles(Params &params, int numBubblesAboveMinRad)
{
  NVTX_RANGE_PUSH_A("BubbleRemoval");

  // Get symbol addresses. These could be cached, but this function is called sufficiently rarely
  // and is already slow enough that fetching them every time isn't a significant impact on performace.
  double *dVolMul   = nullptr;
  double *dTotalVol = nullptr;
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dVolMul), dVolumeMultiplier));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dTotalVol), dTotalVolume));

  CUDA_CALL(cudaMemset(static_cast<void *>(dVolMul), 0, sizeof(double)));

  KERNEL_LAUNCH(calculateRedistributedGasVolume, params.defaultKernelSize, 0, 0, params.ddps[(uint32_t)DDP::TEMP1],
                params.ddps[(uint32_t)DDP::R], params.dips[(uint32_t)DIP::FLAGS], params.state.numBubbles);

  params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP1],
                                                     dTotalVol, params.state.numBubbles);

  int *newIdx = params.dips[(uint32_t)DIP::TEMP1];
  params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, params.dips[(uint32_t)DIP::FLAGS], newIdx,
                               params.state.numBubbles);

  KERNEL_LAUNCH(reorganizeKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles,
                ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, params.dips[(uint32_t)DIP::FLAGS],
                params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::ZP],
                params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::DXDT],
                params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTP],
                params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::DRDT],
                params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::ERROR],
                params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::DZDTO],
                params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::TEMP3],
                params.ddps[(uint32_t)DDP::X0], params.ddps[(uint32_t)DDP::TEMP4], params.ddps[(uint32_t)DDP::Y0],
                params.ddps[(uint32_t)DDP::TEMP5], params.ddps[(uint32_t)DDP::Z0], params.ddps[(uint32_t)DDP::TEMP6],
                params.ddps[(uint32_t)DDP::PATH], params.ddps[(uint32_t)DDP::TEMP7],
                params.ddps[(uint32_t)DDP::DISTANCE], params.ddps[(uint32_t)DDP::TEMP8],
                params.dips[(uint32_t)DIP::WRAP_COUNT_X], params.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                params.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                params.dips[(uint32_t)DIP::WRAP_COUNT_Z], params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(params.ddps[(uint32_t)DDP::XP]), params.state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  params.state.numBubbles  = numBubblesAboveMinRad;
  params.defaultKernelSize = KernelSize(128, params.state.numBubbles);

  KERNEL_LAUNCH(addVolume, params.defaultKernelSize, 0, 0, params.ddps[(uint32_t)DDP::R], params.state.numBubbles);

  NVTX_RANGE_POP();
}

void saveSnapshotToFile(Params &params)
{
  std::stringstream ss;
  ss << params.inputs.snapshotFilename << ".csv." << params.state.numSnapshots;
  std::ofstream file(ss.str().c_str(), std::ios::out);
  if (file.is_open())
  {
    std::vector<double> hostData;
    const size_t numComp = 17;
    hostData.resize(params.state.dataStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), params.deviceDoubleMemory, sizeof(double) * numComp * params.state.dataStride,
                         cudaMemcpyDeviceToHost));

    file << "x,y,z,r,vx,vy,vz,path,dist\n";
    for (size_t i = 0; i < (size_t)params.state.numBubbles; ++i)
    {
      file << hostData[i + 0 * params.state.dataStride];
      file << ",";
      file << hostData[i + 1 * params.state.dataStride];
      file << ",";
      file << hostData[i + 2 * params.state.dataStride];
      file << ",";
      file << hostData[i + 3 * params.state.dataStride];
      file << ",";
      file << hostData[i + 4 * params.state.dataStride];
      file << ",";
      file << hostData[i + 5 * params.state.dataStride];
      file << ",";
      file << hostData[i + 6 * params.state.dataStride];
      file << ",";
      file << hostData[i + 15 * params.state.dataStride];
      file << ",";
      file << hostData[i + 16 * params.state.dataStride];
      file << "\n";
    }

    ++params.state.numSnapshots;
  }
}

double stabilize(Params &params)
{
  // This function integrates only the positions of the bubbles.
  // Gas exchange is not used. This is used for equilibrating the foam.

  double elapsedTime = 0.0;
  double error       = 100000;

  // Energy before stabilization
  KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                params.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y], params.state.interval.z,
                  PBC_Z == 1, params.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y]);

  params.state.energy1 = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);

  for (int i = 0; i < params.inputs.numStepsToRelax; ++i)
  {
    do
    {
      if (NUM_DIM == 3)
      {
        KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                      params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
                      params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::ERROR],
                      params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT],
                      params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                      params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTO],
                      params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDT],
                      params.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                      params.ddps[(uint32_t)DDP::RP], params.state.interval.x, params.state.lbb.x, PBC_X == 1,
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::DXDTP], params.state.interval.y,
                      params.state.lbb.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::DYDTP],
                      params.state.interval.z, params.state.lbb.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::ZP],
                      params.ddps[(uint32_t)DDP::DZDTP]);

        doWallVelocity(params.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, params.state.numBubbles,
                       params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                       params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                       params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
                       params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP], params.state.lbb,
                       params.state.tfr, params);

        KERNEL_LAUNCH(correctKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                      params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                      params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTP],
                      params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDT],
                      params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
                      params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTP]);
      }
      else // Two dimensional case
      {
        KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                      params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
                      params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::TEMP1],
                      params.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT],
                      params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                      params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                      params.ddps[(uint32_t)DDP::RP], params.state.interval.x, params.state.lbb.x, PBC_X == 1,
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::DXDTP], params.state.interval.y,
                      params.state.lbb.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP],
                      params.ddps[(uint32_t)DDP::DYDTP]);

        doWallVelocity(params.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, params.state.numBubbles,
                       params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                       params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                       params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
                       params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP], params.state.lbb,
                       params.state.tfr, params);

        KERNEL_LAUNCH(correctKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                      params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                      params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTP],
                      params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDT],
                      params.ddps[(uint32_t)DDP::DYDTP]);
      }

      doBoundaryWrap(params.defaultKernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, params.state.numBubbles,
                     params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                     params.state.lbb, params.state.tfr, params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Error
      error = params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Max, params.ddps[(uint32_t)DDP::ERROR],
                                                           params.state.numBubbles);

      if (error < params.inputs.errorTolerance && params.state.timeStep < 0.1)
        params.state.timeStep *= 1.9;
      else if (error > params.inputs.errorTolerance)
        params.state.timeStep *= 0.5;

    } while (error > params.inputs.errorTolerance);

    // Update the current values with the calculated predictions
    const size_t numBytesToCopy = 3 * sizeof(double) * params.state.dataStride;
    CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::XP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));

    elapsedTime += params.state.timeStep;

    if (i % 5000 == 0)
      updateCellsAndNeighbors(params);
  }

  // Energy after stabilization
  KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                params.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y], params.state.interval.z,
                  PBC_Z == 1, params.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y]);

  params.state.energy2 = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);

  return elapsedTime;
}

bool integrate(Params &params)
{
  NVTX_RANGE_PUSH_A("Integration function");

  double error              = 100000;
  uint32_t numLoopsDone     = 0;
  int numBubblesAboveMinRad = 0;

  // Get some device global symbol addresses.
  double *totalArea              = nullptr;
  double *totalFreeArea          = nullptr;
  double *totalFreeAreaPerRadius = nullptr;
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&totalArea), dTotalArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&totalFreeArea), dTotalFreeArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&totalFreeAreaPerRadius), dTotalFreeAreaPerRadius));

  do
  {
    NVTX_RANGE_PUSH_A("Integration step");

    if (NUM_DIM == 3)
    {
      // Reset
      KERNEL_LAUNCH(
        resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles, params.ddps[(uint32_t)DDP::DXDTP],
        params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::DRDTP],
        params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                    params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT],
                    params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                    params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::ZP],
                    params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTO],
                    params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::DRDT],
                    params.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, params.velocityStream, params.inputs.fZeroPerMuZero,
                    params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                    params.ddps[(uint32_t)DDP::RP], params.state.interval.x, params.state.lbb.x, PBC_X == 1,
                    params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::DXDTP], params.state.interval.y,
                    params.state.lbb.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::DYDTP],
                    params.state.interval.z, params.state.lbb.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::ZP],
                    params.ddps[(uint32_t)DDP::DZDTP]);
      // Wall velocity
      doWallVelocity(params.pairKernelSize, 0, params.velocityStream, PBC_X == 0, PBC_Y == 0, PBC_Z == 0,
                     params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                     params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                     params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
                     params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP], params.state.lbb,
                     params.state.tfr, params);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * params.state.pairStride));
        int *numNeighbors = params.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, params.pairKernelSize, 0, params.velocityStream,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::DXDTO],
                      params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::DYDTO],
                      params.ddps[(uint32_t)DDP::TEMP3], params.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, params.pairKernelSize, 0, params.velocityStream, params.state.numBubbles,
                      numNeighbors, params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
                      params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::TEMP1],
                      params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::TEMP3],
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                      params.inputs.flowVel, params.inputs.flowTfr, params.inputs.flowLbb);
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, params.defaultKernelSize, 0, params.velocityStream, params.state.numBubbles,
                    params.state.timeStep, params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::XP],
                    params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTP],
                    params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDT],
                    params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::Z],
                    params.ddps[(uint32_t)DDP::DZDT], params.ddps[(uint32_t)DDP::DZDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, params.defaultKernelSize, 0, params.velocityStream,
                    params.state.numBubbles, params.ddps[(uint32_t)DDP::TEMP4], params.ddps[(uint32_t)DDP::PATH],
                    params.ddps[(uint32_t)DDP::DISTANCE], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                    params.ddps[(uint32_t)DDP::X0], params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.interval.x,
                    params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Y0],
                    params.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.state.interval.y, params.ddps[(uint32_t)DDP::ZP],
                    params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::Z0],
                    params.dips[(uint32_t)DIP::WRAP_COUNT_ZP], params.state.interval.z);

      // Boundary wrap
      doBoundaryWrap(params.defaultKernelSize, 0, params.velocityStream, PBC_X == 1, PBC_Y == 1, PBC_Z == 1,
                     params.state.numBubbles, params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                     params.ddps[(uint32_t)DDP::ZP], params.state.lbb, params.state.tfr,
                     params.dips[(uint32_t)DIP::WRAP_COUNT_X], params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_Z], params.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, params.pairKernelSize, 0, params.gasStream, params.state.numBubbles,
                    params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                    params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::DRDTP],
                    params.ddps[(uint32_t)DDP::TEMP1], params.state.interval.x, PBC_X == 1,
                    params.ddps[(uint32_t)DDP::XP], params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP],
                    params.state.interval.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::ZP]);
    }
    else // Two dimensions
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                    params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
                    params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::ERROR],
                    params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                    params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT],
                    params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y],
                    params.ddps[(uint32_t)DDP::DYDT], params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::RP],
                    params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::DRDT], params.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, params.velocityStream, params.inputs.fZeroPerMuZero,
                    params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                    params.ddps[(uint32_t)DDP::RP], params.state.interval.x, params.state.lbb.x, PBC_X == 1,
                    params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::DXDTP], params.state.interval.y,
                    params.state.lbb.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::DYDTP]);
      // Wall velocity
      doWallVelocity(params.pairKernelSize, 0, params.velocityStream, PBC_X == 0, PBC_Y == 0, false,
                     params.state.numBubbles, params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                     params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                     params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::DXDTP],
                     params.ddps[(uint32_t)DDP::DYDTP], params.ddps[(uint32_t)DDP::DZDTP], params.state.lbb,
                     params.state.tfr, params);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * params.state.pairStride));
        int *numNeighbors = params.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, params.pairKernelSize, 0, params.velocityStream,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::DXDTO],
                      params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, params.pairKernelSize, 0, params.velocityStream, params.state.numBubbles,
                      numNeighbors, params.ddps[(uint32_t)DDP::DXDTP], params.ddps[(uint32_t)DDP::DYDTP],
                      params.ddps[(uint32_t)DDP::DZDTP], params.ddps[(uint32_t)DDP::TEMP1],
                      params.ddps[(uint32_t)DDP::TEMP2], params.ddps[(uint32_t)DDP::TEMP3],
                      params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                      params.inputs.flowVel, params.inputs.flowTfr, params.inputs.flowLbb);
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, params.defaultKernelSize, 0, params.velocityStream, params.state.numBubbles,
                    params.state.timeStep, params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::XP],
                    params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDT], params.ddps[(uint32_t)DDP::DXDTP],
                    params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDT],
                    params.ddps[(uint32_t)DDP::DYDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, params.defaultKernelSize, 0, params.velocityStream,
                    params.state.numBubbles, params.ddps[(uint32_t)DDP::TEMP4], params.ddps[(uint32_t)DDP::PATH],
                    params.ddps[(uint32_t)DDP::DISTANCE], params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::X],
                    params.ddps[(uint32_t)DDP::X0], params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.interval.x,
                    params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Y0],
                    params.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.state.interval.y);

      // Boundary wrap
      doBoundaryWrap(params.defaultKernelSize, 0, params.velocityStream, PBC_X == 1, PBC_Y == 1, false,
                     params.state.numBubbles, params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP],
                     params.ddps[(uint32_t)DDP::ZP], params.state.lbb, params.state.tfr,
                     params.dips[(uint32_t)DIP::WRAP_COUNT_X], params.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_Z], params.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     params.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(
        gasExchangeKernel, params.pairKernelSize, 0, params.gasStream, params.state.numBubbles,
        params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::RP],
        params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::TEMP1], params.state.interval.x, PBC_X == 1,
        params.ddps[(uint32_t)DDP::XP], params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::YP]);
    }

    // Free area
    KERNEL_LAUNCH(freeAreaKernel, params.defaultKernelSize, 0, params.gasStream, params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::TEMP1], params.ddps[(uint32_t)DDP::TEMP2],
                  params.ddps[(uint32_t)DDP::TEMP3]);

    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP1],
                                                       totalFreeArea, params.state.numBubbles, params.gasStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP2],
                                                       totalFreeAreaPerRadius, params.state.numBubbles,
                                                       params.gasStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP3],
                                                       totalArea, params.state.numBubbles, params.gasStream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, params.defaultKernelSize, 0, params.gasStream,
                  params.ddps[(uint32_t)DDP::DRDTP], params.ddps[(uint32_t)DDP::RP], params.ddps[(uint32_t)DDP::TEMP1],
                  params.state.numBubbles, params.inputs.kappa, params.inputs.kParameter,
                  params.state.averageSurfaceAreaIn);

    // Radius correct
    KERNEL_LAUNCH(correctKernel, params.defaultKernelSize, 0, params.gasStream, params.state.numBubbles,
                  params.state.timeStep, params.ddps[(uint32_t)DDP::ERROR], params.ddps[(uint32_t)DDP::RP],
                  params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::DRDT], params.ddps[(uint32_t)DDP::DRDTP]);

    // Calculate how many bubbles are below the minimum size.
    // Also take note of maximum radius.
    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, params.defaultKernelSize, 0, params.gasStream,
                  params.state.numBubbles, params.dips[(uint32_t)DIP::FLAGS], params.ddps[(uint32_t)DDP::RP],
                  params.inputs.minRad);

    params.cw.reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, params.dips[(uint32_t)DIP::FLAGS],
                                              params.dips[(uint32_t)DIP::TEMP1], params.state.numBubbles,
                                              params.gasStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, params.ddps[(uint32_t)DDP::RP],
                                                       params.ddps[(uint32_t)DDP::TEMP8], params.state.numBubbles,
                                                       params.gasStream);

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&numBubblesAboveMinRad), params.dips[(uint32_t)DIP::TEMP1],
                              sizeof(int), cudaMemcpyDeviceToHost, params.gasStream));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&params.state.maxBubbleRadius), params.ddps[(uint32_t)DDP::TEMP8],
                              sizeof(double), cudaMemcpyDeviceToHost, params.gasStream));

    // Error
    error = params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Max, params.ddps[(uint32_t)DDP::ERROR],
                                                         params.state.numBubbles);

    if (error < params.inputs.errorTolerance && params.state.timeStep < 0.1)
      params.state.timeStep *= 1.9;
    else if (error > params.inputs.errorTolerance)
      params.state.timeStep *= 0.5;

    ++numLoopsDone;

    NVTX_RANGE_POP();
  } while (error > params.inputs.errorTolerance);

  // Update values
  const size_t numBytesToCopy = 4 * sizeof(double) * params.state.dataStride;

  CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::XP], 2 * numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.ddps[(uint32_t)DDP::PATH], params.ddps[(uint32_t)DDP::TEMP4],
                            sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                            params.state.dataStride * 3 * sizeof(int), cudaMemcpyDeviceToDevice));

  ++params.state.numIntegrationSteps;
  params.state.simulationTime += params.state.timeStep;

  // Delete & reorder
  bool updateNeighbors = params.state.numIntegrationSteps % 5000 == 0;
  if (numBubblesAboveMinRad < params.state.numBubbles)
  {
    deleteSmallBubbles(params, numBubblesAboveMinRad);
    updateNeighbors = true;
  }

  if (updateNeighbors)
    updateCellsAndNeighbors(params);

  bool continueSimulation = params.state.numBubbles > params.inputs.minNumBubbles;
  continueSimulation &= (NUM_DIM == 3)
                          ? params.state.maxBubbleRadius < 0.5 * (params.state.tfr - params.state.lbb).getMinComponent()
                          : true;

  NVTX_RANGE_POP();

  return continueSimulation;
}

void transformPositions(Params &params, bool normalize)
{
  KERNEL_LAUNCH(transformPositionsKernel, params.pairKernelSize, 0, 0, normalize, params.state.numBubbles,
                params.state.lbb, params.state.tfr, params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::Y],
                params.ddps[(uint32_t)DDP::Z]);
}

double calculateVolumeOfBubbles(Params &params)
{
  KERNEL_LAUNCH(calculateVolumes, params.defaultKernelSize, 0, 0, params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::TEMP1], params.state.numBubbles);

  return params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP1],
                                                      params.state.numBubbles);
}

void deinit(Params &params)
{
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(static_cast<void *>(params.deviceDoubleMemory)));
  CUDA_CALL(cudaFree(static_cast<void *>(params.deviceIntMemory)));

  CUDA_CALL(cudaStreamDestroy(params.velocityStream));
  CUDA_CALL(cudaStreamDestroy(params.gasStream));
}

double getSimulationBoxVolume(Params &params)
{
  dvec temp = params.state.tfr - params.state.lbb;
  return (NUM_DIM == 3) ? temp.x * temp.y * temp.z : temp.x * temp.y;
}

#define JSON_READ(i, j, arg) \
  i.arg = j[#arg];           \
  std::cout << #arg << ": " << i.arg << std::endl

void readInputs(Params &params, const char *inputFileName, ivec &bubblesPerDim)
{
  std::cout << "Reading inputs from file \"" << inputFileName << "\"" << std::endl;
  nlohmann::json j;
  std::fstream file(inputFileName, std::ios::in);

  if (file.is_open())
  {
    file >> j;
    SimulationInputs &inputs = params.inputs;

    JSON_READ(inputs, j, phiTarget);
    JSON_READ(inputs, j, muZero);
    JSON_READ(inputs, j, sigmaZero);
    JSON_READ(inputs, j, avgRad);
    JSON_READ(inputs, j, stdDevRad);
    JSON_READ(inputs, j, errorTolerance);
    JSON_READ(inputs, j, timeStepIn);
    JSON_READ(inputs, j, rngSeed);
    JSON_READ(inputs, j, numBubblesPerCell);
    JSON_READ(inputs, j, snapshotFilename);
    JSON_READ(inputs, j, numStepsToRelax);
    JSON_READ(inputs, j, maxDeltaEnergy);
    JSON_READ(inputs, j, kParameter);
    JSON_READ(inputs, j, numBubblesIn);
    JSON_READ(inputs, j, kappa);
    JSON_READ(inputs, j, minNumBubbles);
    JSON_READ(inputs, j, dataFilename);
    JSON_READ(inputs, j, boxRelDim);
    JSON_READ(inputs, j, flowLbb);
    JSON_READ(inputs, j, flowTfr);
    JSON_READ(inputs, j, flowVel);

    assert(inputs.muZero > 0);
    assert(inputs.boxRelDim.x > 0);
    assert(inputs.boxRelDim.y > 0);
    assert(inputs.boxRelDim.z > 0);

    inputs.fZeroPerMuZero    = inputs.sigmaZero * inputs.avgRad / inputs.muZero;
    inputs.minRad            = 0.1 * inputs.avgRad;
    inputs.timeScalingFactor = inputs.kParameter / (inputs.avgRad * inputs.avgRad);
    inputs.flowVel *= inputs.fZeroPerMuZero;
  }
  else
    throw std::runtime_error("Couldn't open input file!");

  // First calculate the size of the box and the starting number of bubbles
  dvec relDim   = params.inputs.boxRelDim;
  relDim        = relDim / relDim.x;
  const float d = 2 * params.inputs.avgRad;
  float x       = params.inputs.numBubblesIn * d * d / relDim.y;
  bubblesPerDim = ivec(0, 0, 0);

  if (NUM_DIM == 3)
  {
    x                       = x * d / relDim.z;
    x                       = std::cbrt(x);
    relDim                  = relDim * x;
    bubblesPerDim           = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), std::ceil(relDim.z / d));
    params.state.numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
  }
  else
  {
    x                       = std::sqrt(x);
    relDim                  = relDim * x;
    bubblesPerDim           = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
    params.state.numBubbles = bubblesPerDim.x * bubblesPerDim.y;
  }

  params.state.tfr      = d * bubblesPerDim.asType<double>() + params.state.lbb;
  params.state.interval = params.state.tfr - params.state.lbb;
  params.inputs.flowTfr = params.state.interval * params.inputs.flowTfr + params.state.lbb;
  params.inputs.flowLbb = params.state.interval * params.inputs.flowLbb + params.state.lbb;
  params.state.timeStep = params.inputs.timeStepIn;

  // Determine the maximum number of Morton numbers for the simulation box
  dim3 gridDim         = getGridSize(params);
  const int maxGridDim = gridDim.x > gridDim.y ? (gridDim.x > gridDim.z ? gridDim.x : gridDim.z)
                                               : (gridDim.y > gridDim.z ? gridDim.y : gridDim.z);
  int maxNumCells = 1;
  while (maxNumCells < maxGridDim)
    maxNumCells = maxNumCells << 1;

  if (NUM_DIM == 3)
    maxNumCells = maxNumCells * maxNumCells * maxNumCells;
  else
    maxNumCells = maxNumCells * maxNumCells;

  params.state.maxNumCells = maxNumCells;

  std::cout << "Maximum (theoretical) number of cells: " << params.state.maxNumCells
            << ", actual grid dimensions: " << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << std::endl;
}
#undef JSON_READ

void commonSetup(Params &params)
{
  params.defaultKernelSize = KernelSize(128, params.state.numBubbles);

  // Streams
  CUDA_ASSERT(cudaStreamCreate(&params.velocityStream));
  CUDA_ASSERT(cudaStreamCreate(&params.gasStream));

  printRelevantInfoOfCurrentDevice();

  std::cout << "Reserving device memory to hold data." << std::endl;

  // Calculate the length of 'rows'. Will be divisible by 32, as that's the warp size.
  params.state.dataStride =
    params.state.numBubbles + !!(params.state.numBubbles % 32) * (32 - params.state.numBubbles % 32);

  // Doubles
  params.state.memReqD = sizeof(double) * (uint64_t)params.state.dataStride * (uint64_t)DDP::NUM_VALUES;
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceDoubleMemory), params.state.memReqD));

  for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
    params.ddps[i] = params.deviceDoubleMemory + i * params.state.dataStride;

  // Integers
  // 32 is just a guess, and roughly it seems to hold true with 3D sim.
  const uint32_t avgNumNeighbors = 32;
  params.state.pairStride        = avgNumNeighbors * params.state.dataStride;

  params.state.memReqI = sizeof(int) * (uint64_t)params.state.dataStride *
                         ((uint64_t)DIP::PAIR1 + avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.deviceIntMemory), params.state.memReqI));

  for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
    params.dips[i] = params.deviceIntMemory + i * params.state.dataStride;

  uint32_t j = 0;
  for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
    params.dips[i] = params.dips[(uint32_t)DIP::PAIR1] + avgNumNeighbors * ++j * params.state.dataStride;

  std::cout << "Memory requirement for data:\n\tdouble: " << params.state.memReqD
            << " bytes\n\tint: " << params.state.memReqI
            << " bytes\n\ttotal: " << params.state.memReqI + params.state.memReqD << " bytes" << std::endl;
}

void generateStartingData(Params &params, ivec bubblesPerDim)
{
  std::cout << "Starting to generate data for bubbles." << std::endl;
  const int rngSeed      = params.inputs.rngSeed;
  const double avgRad    = params.inputs.avgRad;
  const double stdDevRad = params.inputs.stdDevRad;

  curandGenerator_t generator;
  CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
  if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, params.ddps[(uint32_t)DDP::Z], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.ddps[(uint32_t)DDP::X], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.ddps[(uint32_t)DDP::Y], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.ddps[(uint32_t)DDP::RP], params.state.numBubbles));
  CURAND_CALL(
    curandGenerateNormalDouble(generator, params.ddps[(uint32_t)DDP::R], params.state.numBubbles, avgRad, stdDevRad));
  CURAND_CALL(curandDestroyGenerator(generator));

  KERNEL_LAUNCH(assignDataToBubbles, params.defaultKernelSize, 0, 0, params.ddps[(uint32_t)DDP::X],
                params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::XP],
                params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP], params.ddps[(uint32_t)DDP::R],
                params.ddps[(uint32_t)DDP::RP], params.dips[(uint32_t)DIP::FLAGS], bubblesPerDim, params.state.tfr,
                params.state.lbb, avgRad, params.inputs.minRad, params.state.numBubbles);

  params.state.averageSurfaceAreaIn = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::RP], params.state.numBubbles, 0);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.ddps[(uint32_t)DDP::RP]),
                            static_cast<void *>(params.ddps[(uint32_t)DDP::R]),
                            sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));

  std::cout << "Deleting small bubbles and updating neighbor lists." << std::endl;
  const int numBubblesAboveMinRad = params.cw.reduce<int, int *, int *>(
    &cub::DeviceReduce::Sum, params.dips[(uint32_t)DIP::FLAGS], params.state.numBubbles);
  if (numBubblesAboveMinRad < params.state.numBubbles)
    deleteSmallBubbles(params, numBubblesAboveMinRad);

  params.state.maxBubbleRadius = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Max, params.ddps[(uint32_t)DDP::R], params.state.numBubbles);

  updateCellsAndNeighbors(params);

  // Calculate some initial values which are needed
  // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
  KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::DZDTO],
                params.ddps[(uint32_t)DDP::DRDTO], params.ddps[(uint32_t)DDP::DISTANCE],
                params.ddps[(uint32_t)DDP::PATH]);

  std::cout << "Calculating some initial values as a part of setup." << std::endl;
  if (NUM_DIM == 3)
  {
    KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.state.interval.x, params.state.lbb.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::DXDTO], params.state.interval.y, params.state.lbb.y, PBC_Y == 1,
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDTO], params.state.interval.z,
                  params.state.lbb.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(eulerKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::DYDTO], params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDTO]);

    doBoundaryWrap(params.defaultKernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, params.state.numBubbles,
                   params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                   params.state.lbb, params.state.tfr, params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
                  params.ddps[(uint32_t)DDP::DZDTO], params.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.state.interval.x, params.state.lbb.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::DXDTO], params.state.interval.y, params.state.lbb.y, PBC_Y == 1,
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDTO], params.state.interval.z,
                  params.state.lbb.z, PBC_Z == 1, params.ddps[(uint32_t)DDP::Z], params.ddps[(uint32_t)DDP::DZDTO]);
  }
  else
  {
    KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.state.interval.x, params.state.lbb.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::DXDTO], params.state.interval.y, params.state.lbb.y, PBC_Y == 1,
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDTO]);

    KERNEL_LAUNCH(eulerKernel, params.defaultKernelSize, 0, 0, params.state.numBubbles, params.state.timeStep,
                  params.ddps[(uint32_t)DDP::X], params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::Y],
                  params.ddps[(uint32_t)DDP::DYDTO]);

    doBoundaryWrap(params.defaultKernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, false, params.state.numBubbles,
                   params.ddps[(uint32_t)DDP::XP], params.ddps[(uint32_t)DDP::YP], params.ddps[(uint32_t)DDP::ZP],
                   params.state.lbb, params.state.tfr, params.dips[(uint32_t)DIP::WRAP_COUNT_X],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                   params.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                  params.ddps[(uint32_t)DDP::DXDTO], params.ddps[(uint32_t)DDP::DYDTO],
                  params.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, params.pairKernelSize, 0, 0, params.inputs.fZeroPerMuZero,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.state.interval.x, params.state.lbb.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.ddps[(uint32_t)DDP::DXDTO], params.state.interval.y, params.state.lbb.y, PBC_Y == 1,
                  params.ddps[(uint32_t)DDP::Y], params.ddps[(uint32_t)DDP::DYDTO]);
  }
}

void deserializeState(Params &params, const char *inputFileName)
{
  // Deserialize the saved state of the simulation.
}

void deserializeData(Params &params, const char *inputFileName)
{
  // Deserialize the data from the binary and copy it to the device.
  // When serializing data, already 'remove' the useless bits at the end of each 'row'.
}

void initializeFromJson(const char *inputFileName, Params &params, std::stringstream &dataStream)
{
  // Initialize everything, starting with an input .json file.
  // The end state of this function is 'prepared state' that can then be used immediately to run the integration loop.

  std::cout << "\n=====\nSetup\n=====" << std::endl;
  ivec bubblesPerDim = ivec(0, 0, 0);
  readInputs(params, inputFileName, bubblesPerDim);
  commonSetup(params);
  generateStartingData(params, bubblesPerDim);
  saveSnapshotToFile(params); // 0

  std::cout << "Letting bubbles settle after they've been created and before "
               "scaling or stabilization."
            << std::endl;

  stabilize(params);
  saveSnapshotToFile(params); // 1

  const double bubbleVolume = calculateVolumeOfBubbles(params);

  std::cout << "Volume ratios: current: " << bubbleVolume / getSimulationBoxVolume(params)
            << ", target: " << params.inputs.phiTarget << "\nScaling the simulation box." << std::endl;

  transformPositions(params, true);

  dvec relativeSize     = params.inputs.boxRelDim;
  relativeSize.z        = (NUM_DIM == 2) ? 1 : relativeSize.z;
  double t              = bubbleVolume / (params.inputs.phiTarget * relativeSize.x * relativeSize.y * relativeSize.z);
  t                     = (NUM_DIM == 3) ? std::cbrt(t) : std::sqrt(t);
  params.state.tfr      = dvec(t, t, t) * relativeSize;
  params.state.interval = params.state.tfr - params.state.lbb;

  transformPositions(params, false);
  saveSnapshotToFile(params); // 2

  std::cout << "Volume ratios: current: " << bubbleVolume / getSimulationBoxVolume(params)
            << ", target: " << params.inputs.phiTarget
            << "\n\n=============\nStabilization\n=============" << std::endl;

  int numSteps       = 0;
  const int failsafe = 500;
  std::cout << "#steps\tdE/t\te1\te2" << std::endl;
  while (true)
  {
    double time        = stabilize(params);
    double deltaEnergy = std::abs(params.state.energy2 - params.state.energy1) / time;
    deltaEnergy *= 0.5 * params.inputs.sigmaZero;

    if (deltaEnergy < params.inputs.maxDeltaEnergy)
    {
      std::cout << "Final delta energy " << deltaEnergy << " after " << (numSteps + 1) * params.inputs.numStepsToRelax
                << " steps."
                << " Energy before: " << params.state.energy1 << ", energy after: " << params.state.energy2
                << ", time: " << time * params.inputs.timeScalingFactor << std::endl;
      break;
    }
    else if (numSteps > failsafe)
    {
      std::cout << "Over " << failsafe * params.inputs.numStepsToRelax
                << " steps taken and required delta energy not reached."
                << " Check parameters." << std::endl;
      break;
    }
    else
    {
      std::cout << (numSteps + 1) * params.inputs.numStepsToRelax << "\t" << deltaEnergy << "\t" << params.state.energy1
                << "\t" << params.state.energy2 << std::endl;
    }

    ++numSteps;
  }

  saveSnapshotToFile(params); // 3

  // Set starting positions and reset wrap counts to 0
  const size_t numBytesToCopy = 3 * sizeof(double) * params.state.dataStride;
  CUDA_CALL(cudaMemcpy(params.ddps[(uint32_t)DDP::X0], params.ddps[(uint32_t)DDP::X], numBytesToCopy,
                       cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemset(params.dips[(uint32_t)DIP::WRAP_COUNT_X], 0, 6 * params.state.dataStride * sizeof(int)));

  // Calculate the energy at starting positions
  KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                params.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
  {
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y], params.state.interval.z,
                  PBC_Z == 1, params.ddps[(uint32_t)DDP::Z]);
  }
  else
  {
    KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2], params.ddps[(uint32_t)DDP::R],
                  params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x, PBC_X == 1, params.ddps[(uint32_t)DDP::X],
                  params.state.interval.y, PBC_Y == 1, params.ddps[(uint32_t)DDP::Y]);
  }

  params.state.energy1 = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Sum, params.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);
  params.state.simulationTime      = 0.0;
  params.state.timesPrinted        = 1;
  params.state.numIntegrationSteps = 0;
}

void initializeFromBinary(const char *inputFileName, Params &params, std::stringstream &dataStream)
{
  // This function initializes the simulation state from a binary dump.
  // The end state of this function is 'prepared state' that can then be used immediately to run the integration loop.
  deserializeState(params, inputFileName);
  commonSetup(params);
  deserializeData(params, inputFileName);
}

} // namespace

namespace cubble
{
void run(const char *inputFileName)
{
  Params params;
  std::stringstream dataStream;

  initializeFromJson(inputFileName, params, dataStream);
  // initializeFromBinary(inputFileName, params, dataStream);

  std::cout << "\n==========\nIntegration\n==========" << std::endl;
  bool continueIntegration = true;
  std::cout << "T\tphi\tR\t#b\tdE\t\t#steps\t#pairs" << std::endl;
  while (continueIntegration)
  {
    continueIntegration = integrate(params);
    CUDA_PROFILER_START(params.state.numIntegrationSteps == 2000);
    CUDA_PROFILER_STOP(params.state.numIntegrationSteps == 2200, continueIntegration);

    const double scaledTime = params.state.simulationTime * params.inputs.timeScalingFactor;
    if ((int)scaledTime >= params.state.timesPrinted)
    {
      // Calculate total energy
      KERNEL_LAUNCH(resetKernel, params.defaultKernelSize, 0, 0, 0.0, params.state.numBubbles,
                    params.ddps[(uint32_t)DDP::TEMP4]);

      if (NUM_DIM == 3)
        KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                      params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x,
                      PBC_X == 1, params.ddps[(uint32_t)DDP::X], params.state.interval.y, PBC_Y == 1,
                      params.ddps[(uint32_t)DDP::Y], params.state.interval.z, PBC_Z == 1,
                      params.ddps[(uint32_t)DDP::Z]);
      else
        KERNEL_LAUNCH(potentialEnergyKernel, params.pairKernelSize, 0, 0, params.state.numBubbles,
                      params.dips[(uint32_t)DIP::PAIR1], params.dips[(uint32_t)DIP::PAIR2],
                      params.ddps[(uint32_t)DDP::R], params.ddps[(uint32_t)DDP::TEMP4], params.state.interval.x,
                      PBC_X == 1, params.ddps[(uint32_t)DDP::X], params.state.interval.y, PBC_Y == 1,
                      params.ddps[(uint32_t)DDP::Y]);

      auto getSum = [](double *p, Params &params) -> double {
        return params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, p, params.state.numBubbles);
      };

      auto getAvg = [getSum](double *p, Params &params) -> double {
        return getSum(p, params) / params.state.numBubbles;
      };

      params.state.energy2        = getSum(params.ddps[(uint32_t)DDP::TEMP4], params);
      const double dE             = (params.state.energy2 - params.state.energy1) / params.state.energy2;
      const double relativeRadius = getAvg(params.ddps[(uint32_t)DDP::R], params) / params.inputs.avgRad;

      // Add values to data stream
      dataStream << (int)scaledTime << " " << relativeRadius << " "
                 << params.state.maxBubbleRadius / params.inputs.avgRad << " " << params.state.numBubbles << " "
                 << getAvg(params.ddps[(uint32_t)DDP::PATH], params) << " "
                 << getAvg(params.ddps[(uint32_t)DDP::DISTANCE], params) << " " << dE << "\n";

      // Print some values
      std::cout << (int)scaledTime << "\t" << calculateVolumeOfBubbles(params) / getSimulationBoxVolume(params) << "\t"
                << relativeRadius << "\t" << params.state.numBubbles << "\t" << dE << "\t"
                << params.state.numStepsInTimeStep << "\t" << params.state.numPairs << std::endl;

      // Only write snapshots when t* is a power of 2.
      if ((params.state.timesPrinted & (params.state.timesPrinted - 1)) == 0)
        saveSnapshotToFile(params);

      ++params.state.timesPrinted;
      params.state.numStepsInTimeStep = 0;
      params.state.energy1            = params.state.energy2;
    }

    ++params.state.numStepsInTimeStep;
  }

  // Only save when actually ending, and not due to time running out
  saveSnapshotToFile(params);

  // Append when continued
  std::ofstream file(params.inputs.dataFilename);
  file << dataStream.str() << std::endl;

  deinit(params);
}
} // namespace cubble
