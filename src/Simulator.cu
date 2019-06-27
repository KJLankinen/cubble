// -*- C++ -*-

#include "Kernels.cuh"
#include "Macros.h"
#include "Simulator.cuh"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <algorithm>
#include <chrono>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <iostream>
#include <nvToolsExt.h>
#include <sstream>
#include <vector>

namespace cubble
{

void Simulator::run(const char *inputFileName, const char *outputFileName)
{
  auto getVolumeOfBubbles = [this]() -> double {
    KernelSize kernelSize(128, numBubbles);

    KERNEL_LAUNCH(calculateVolumes, kernelSize, 0, 0, adp.r, adp.dummy1, numBubbles);

    return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1,
                                                          numBubbles);
  };

  properties = Env(inputFileName, outputFileName);
  properties.readParameters();

  std::cout << "======\nSetup\n======" << std::endl;
  {
    setup();
    saveSnapshotToFile();

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;
    stabilize();
    saveSnapshotToFile();

    const double phiTarget = properties.getPhiTarget();
    double bubbleVolume    = getVolumeOfBubbles();
    double phi             = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;
    transformPositions(true);
    dvec relativeSize = properties.getBoxRelativeDimensions();
    relativeSize.z    = (NUM_DIM == 2) ? 1 : relativeSize.z;
    double t =
      getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z);
    t = (NUM_DIM == 3) ? std::cbrt(t) : std::sqrt(t);
    properties.setTfr(dvec(t, t, t) * relativeSize);
    transformPositions(false);

    phi = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    saveSnapshotToFile();
  }

  std::cout << "=============\nStabilization\n=============" << std::endl;
  {
    int numSteps       = 0;
    const int failsafe = 500;

    std::cout << "#steps\tdE/t\te1\te2" << std::endl;
    while (true)
    {
      double time        = stabilize();
      double deltaEnergy = std::abs(energy2 - energy1) / time;
      deltaEnergy *= 0.5 * properties.getSigmaZero();

      if (deltaEnergy < properties.getMaxDeltaEnergy())
      {
        std::cout << "Final delta energy " << deltaEnergy << " after "
                  << (numSteps + 1) * properties.getNumStepsToRelax() << " steps."
                  << " Energy before: " << energy1 << ", energy after: " << energy2
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
        std::cout << (numSteps + 1) * properties.getNumStepsToRelax() << "\t" << deltaEnergy << "\t"
                  << energy1 << "\t" << energy2 << std::endl;
      }

      ++numSteps;
    }

    saveSnapshotToFile();
  }

  std::cout << "==========\nSimulation\n==========" << std::endl;
  std::stringstream dataStream;
  {
    // Set starting positions and reset wrapMultipliers to 0
    const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
    CUDA_CALL(cudaMemcpy(adp.x0, adp.x, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemset(wrapMultipliers.get(), 0, wrapMultipliers.getSizeInBytes()));

    simulationTime      = 0;
    int timesPrinted    = 1;
    uint32_t numSteps   = 0;
    const dvec interval = properties.getTfr() - properties.getLbb();

    KernelSize kernelSize(128, numBubbles);

    // Calculate the energy at simulation start
    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dummy4);

    if (NUM_DIM == 3)
    {
      KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                    pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x,
                    interval.y, PBC_Y == 1, adp.y, interval.z, PBC_Z == 1, adp.z);
    }
    else
    {
      KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                    pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x,
                    interval.y, PBC_Y == 1, adp.y);
    }

    energy1 = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4,
                                                             numBubbles);

    // Start the simulation proper
    bool continueIntegration = true;
    int numTotalSteps        = 0;
    std::cout << "T\tR\t#b\tdE\t#steps" << std::endl;
    while (continueIntegration)
    {
      continueIntegration = integrate();
      CUDA_PROFILER_START(numTotalSteps == 0);
      CUDA_PROFILER_STOP(numTotalSteps == 9000, continueIntegration);

      // The if clause contains many slow operations, but it's only done
      // very few times relative to the entire run time, so it should not
      // have a huge cost. Roughly 6e4-1e5 integration steps are taken for each
      // time step
      // and the if clause is executed once per time step.
      const double scaledTime = simulationTime * properties.getTimeScalingFactor();
      if ((int)scaledTime >= timesPrinted)
      {
        // Calculate total energy
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dummy4);

        if (NUM_DIM == 3)
          KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                        pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x,
                        interval.y, PBC_Y == 1, adp.y, interval.z, PBC_Z == 1, adp.z);
        else
          KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                        pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x,
                        interval.y, PBC_Y == 1, adp.y);

        energy2 = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                 adp.dummy4, numBubbles);
        const double dE = (energy2 - energy1) / energy2;

        // Add values to data stream
        double relativeRadius = getAverageProperty(adp.r) / properties.getAvgRad();
        dataStream << scaledTime << " " << relativeRadius << " "
                   << maxBubbleRadius / properties.getAvgRad() << " " << numBubbles << " "
                   << getAverageProperty(adp.s) << " " << getAverageProperty(adp.d) << " " << dE
                   << "\n";

        // Print some values
        std::cout << scaledTime << "\t" << relativeRadius << "\t" << numBubbles << "\t" << dE
                  << "\t" << numSteps << std::endl;

        // Only write snapshots when t* is a power of 2.
        if ((timesPrinted & (timesPrinted - 1)) == 0)
          saveSnapshotToFile();

        ++timesPrinted;
        numSteps = 0;
        energy1  = energy2;
      }

      ++numSteps;
      ++numTotalSteps;
    }
  }

  std::ofstream file(properties.getDataFilename());
  file << dataStream.str() << std::endl;

  deinit();
}

void Simulator::setup()
{
  dvec relDim        = properties.getBoxRelativeDimensions();
  relDim             = relDim / relDim.x;
  const float d      = 2 * properties.getAvgRad();
  float x            = properties.getNumBubbles() * d * d / relDim.y;
  ivec bubblesPerDim = ivec(0, 0, 0);

  if (NUM_DIM == 3)
  {
    x             = x * d / relDim.z;
    x             = std::cbrt(x);
    relDim        = relDim * x;
    bubblesPerDim = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), std::ceil(relDim.z / d));
    numBubbles    = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
  }
  else
  {
    x             = std::sqrt(x);
    relDim        = relDim * x;
    bubblesPerDim = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
    numBubbles    = bubblesPerDim.x * bubblesPerDim.y;
  }

  properties.setTfr(d * bubblesPerDim.asType<double>() + properties.getLbb());
  dvec interval = properties.getTfr() - properties.getLbb();
  properties.setFlowTfr(interval * properties.getFlowTfr() + properties.getLbb());
  properties.setFlowLbb(interval * properties.getFlowLbb() + properties.getLbb());

  cubWrapper = std::make_shared<CubWrapper>(numBubbles * sizeof(double));

  dataStride = numBubbles + !!(numBubbles % 32) * (32 - numBubbles % 32);
  CUDA_ASSERT(
    cudaMalloc(reinterpret_cast<void **>(&deviceData), sizeof(double) * dataStride * numAliases));
  adp.x      = deviceData;
  adp.y      = deviceData + 1 * dataStride;
  adp.z      = deviceData + 2 * dataStride;
  adp.r      = deviceData + 3 * dataStride;
  adp.dxdt   = deviceData + 4 * dataStride;
  adp.dydt   = deviceData + 5 * dataStride;
  adp.dzdt   = deviceData + 6 * dataStride;
  adp.drdt   = deviceData + 7 * dataStride;
  adp.dxdtO  = deviceData + 8 * dataStride;
  adp.dydtO  = deviceData + 9 * dataStride;
  adp.dzdtO  = deviceData + 10 * dataStride;
  adp.drdtO  = deviceData + 11 * dataStride;
  adp.x0     = deviceData + 12 * dataStride;
  adp.y0     = deviceData + 13 * dataStride;
  adp.z0     = deviceData + 14 * dataStride;
  adp.s      = deviceData + 15 * dataStride;
  adp.d      = deviceData + 16 * dataStride;
  adp.xP     = deviceData + 17 * dataStride;
  adp.yP     = deviceData + 18 * dataStride;
  adp.zP     = deviceData + 19 * dataStride;
  adp.rP     = deviceData + 20 * dataStride;
  adp.dxdtP  = deviceData + 21 * dataStride;
  adp.dydtP  = deviceData + 22 * dataStride;
  adp.dzdtP  = deviceData + 23 * dataStride;
  adp.drdtP  = deviceData + 24 * dataStride;
  adp.error  = deviceData + 25 * dataStride;
  adp.dummy1 = deviceData + 26 * dataStride;
  adp.dummy2 = deviceData + 27 * dataStride;
  adp.dummy3 = deviceData + 28 * dataStride;
  adp.dummy4 = deviceData + 29 * dataStride;
  adp.dummy5 = deviceData + 30 * dataStride;
  adp.dummy6 = deviceData + 31 * dataStride;
  adp.dummy7 = deviceData + 32 * dataStride;
  adp.dummy8 = deviceData + 33 * dataStride;

  // Determine the maximum number of Morton numbers for the cell
  dim3 gridDim         = getGridSize();
  const int maxGridDim = gridDim.x > gridDim.y ? (gridDim.x > gridDim.z ? gridDim.x : gridDim.z)
                                               : (gridDim.y > gridDim.z ? gridDim.y : gridDim.z);
  maxNumCells = 1;
  while (maxNumCells < maxGridDim)
    maxNumCells = maxNumCells << 1;

  if (NUM_DIM == 3)
    maxNumCells = maxNumCells * maxNumCells * maxNumCells;
  else
    maxNumCells = maxNumCells * maxNumCells;

  std::cout << "Morton: " << maxNumCells << ", " << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << std::endl;

  aboveMinRadFlags  = DeviceArray<int>(dataStride, 2u);
  bubbleCellIndices = DeviceArray<int>(dataStride, 4u);
  pairs             = DeviceArray<int>(8 * dataStride, 4u);
  wrapMultipliers   = DeviceArray<int>(dataStride, 6);
  cellData          = DeviceArray<int>(maxNumCells, (size_t)CellProperty::NUM_VALUES);

  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfa), dTotalFreeArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfapr), dTotalFreeAreaPerRadius));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&mbpc), dMaxBubblesPerCell));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dvm), dVolumeMultiplier));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtv), dTotalVolume));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&np), dNumPairs));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dir), dInvRho));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dta), dTotalArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dasai), dAverageSurfaceAreaIn));

  CUDA_ASSERT(cudaStreamCreateWithFlags(&nonBlockingStream, cudaStreamNonBlocking));
  CUDA_ASSERT(cudaStreamCreate(&velocityStream));
  CUDA_ASSERT(cudaStreamCreate(&gasExchangeStream));

  CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent1, cudaEventBlockingSync));
  CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent2, cudaEventBlockingSync));

  pinnedInt    = PinnedHostArray<int>(1);
  pinnedDouble = PinnedHostArray<double>(3);

  printRelevantInfoOfCurrentDevice();

  pairKernelSize.block = dim3(128, 1, 1);
  pairKernelSize.grid  = dim3(256, 1, 1);

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
    CURAND_CALL(curandGenerateUniformDouble(generator, adp.z, numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, adp.x, numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, adp.y, numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, adp.rP, numBubbles));
  CURAND_CALL(curandGenerateNormalDouble(generator, adp.r, numBubbles, avgRad, stdDevRad));

  CURAND_CALL(curandDestroyGenerator(generator));

  KernelSize kernelSize(128, numBubbles);

  KERNEL_LAUNCH(assignDataToBubbles, kernelSize, 0, 0, adp.x, adp.y, adp.z, adp.xP, adp.yP, adp.zP,
                adp.r, adp.rP, aboveMinRadFlags.getRowPtr(0), bubblesPerDim, tfr, lbb, avgRad,
                properties.getMinRad(), numBubbles);

  cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.rP, dasai,
                                                       numBubbles, 0);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.rP), static_cast<void *>(adp.r),
                            sizeof(double) * dataStride, cudaMemcpyDeviceToDevice, 0));

  const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(
    &cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), numBubbles);
  if (numBubblesAboveMinRad < numBubbles)
    deleteSmallBubbles(numBubblesAboveMinRad);

  updateCellsAndNeighbors();

  // Calculate some initial values which are needed
  // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method

  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtO, adp.dydtO, adp.dzdtO,
                adp.drdtO, adp.d, adp.s);

  std::cout << "Calculating some initial values as a part of setup." << std::endl;

  if (NUM_DIM == 3)
  {

    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r, interval.x, lbb.x, PBC_X == 1,
                  adp.x, adp.dxdtO, interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO, interval.z,
                  lbb.z, PBC_Z == 1, adp.z, adp.dzdtO);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.x, adp.dxdtO, adp.y,
                  adp.dydtO, adp.z, adp.dzdtO);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles, adp.xP, adp.yP,
                   adp.zP, lbb, tfr, wrapMultipliers.getRowPtr(3), wrapMultipliers.getRowPtr(0),
                   wrapMultipliers.getRowPtr(4), wrapMultipliers.getRowPtr(1),
                   wrapMultipliers.getRowPtr(5), wrapMultipliers.getRowPtr(2));

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtO, adp.dydtO, adp.dzdtO,
                  adp.drdtO);

    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r, interval.x, lbb.x, PBC_X == 1,
                  adp.x, adp.dxdtO, interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO, interval.z,
                  lbb.z, PBC_Z == 1, adp.z, adp.dzdtO);
  }
  else
  {
    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r, interval.x, lbb.x, PBC_X == 1,
                  adp.x, adp.dxdtO, interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.x, adp.dxdtO, adp.y,
                  adp.dydtO);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, false, numBubbles, adp.xP, adp.yP,
                   adp.zP, lbb, tfr, wrapMultipliers.getRowPtr(3), wrapMultipliers.getRowPtr(0),
                   wrapMultipliers.getRowPtr(4), wrapMultipliers.getRowPtr(1),
                   wrapMultipliers.getRowPtr(5), wrapMultipliers.getRowPtr(2));

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtO, adp.dydtO, adp.drdtO);

    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r, interval.x, lbb.x, PBC_X == 1,
                  adp.x, adp.dxdtO, interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO);
  }
}

void Simulator::deinit()
{
  saveSnapshotToFile();
  properties.writeParameters();

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(static_cast<void *>(deviceData)));

  CUDA_CALL(cudaStreamDestroy(nonBlockingStream));
  CUDA_CALL(cudaStreamDestroy(velocityStream));
  CUDA_CALL(cudaStreamDestroy(gasExchangeStream));

  CUDA_CALL(cudaEventDestroy(blockingEvent1));
  CUDA_CALL(cudaEventDestroy(blockingEvent2));
}

double Simulator::stabilize()
{
  // This function integrates only the positions of the bubbles.
  // Gas exchange is not used. This is used for equilibrating the foam.

  KernelSize kernelSize(128, numBubbles);

  const dvec tfr      = properties.getTfr();
  const dvec lbb      = properties.getLbb();
  const dvec interval = tfr - lbb;
  double elapsedTime  = 0.0;
  double timeStep     = properties.getTimeStep();
  double error        = 100000;

  // Energy before stabilization
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dummy4);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                  pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x, interval.y,
                  PBC_Y == 1, adp.y, interval.z, PBC_Z == 1, adp.z);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                  pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x, interval.y,
                  PBC_Y == 1, adp.y);

  cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, dtfapr,
                                                       numBubbles);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&pinnedDouble.get()[1]),
                            static_cast<void *>(dtfapr), sizeof(double), cudaMemcpyDeviceToHost,
                            0));

  for (int i = 0; i < properties.getNumStepsToRelax(); ++i)
  {
    do
    {
      if (NUM_DIM == 3)
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtP, adp.dydtP,
                      adp.dzdtP, adp.error, adp.dummy1, adp.dummy2);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.xP, adp.x,
                      adp.dxdt, adp.dxdtO, adp.yP, adp.y, adp.dydt, adp.dydtO, adp.zP, adp.z,
                      adp.dzdt, adp.dzdtO);

        KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, interval.x, lbb.x, PBC_X == 1,
                      adp.xP, adp.dxdtP, interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP,
                      interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP);

        doWallVelocity(pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, numBubbles,
                       pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.xP, adp.yP, adp.zP,
                       adp.dxdtP, adp.dydtP, adp.dzdtP, lbb, tfr);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.error, adp.xP,
                      adp.x, adp.dxdt, adp.dxdtP, adp.yP, adp.y, adp.dydt, adp.dydtP, adp.zP, adp.z,
                      adp.dzdt, adp.dzdtP);
      }
      else // Two dimensional case
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtP, adp.dydtP,
                      adp.error, adp.dummy1, adp.dummy2);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.xP, adp.x,
                      adp.dxdt, adp.dxdtO, adp.yP, adp.y, adp.dydt, adp.dydtO);

        KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, interval.x, lbb.x, PBC_X == 1,
                      adp.xP, adp.dxdtP, interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP);

        doWallVelocity(pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, numBubbles,
                       pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.xP, adp.yP, adp.zP,
                       adp.dxdtP, adp.dydtP, adp.dzdtP, lbb, tfr);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.error, adp.xP,
                      adp.x, adp.dxdt, adp.dxdtP, adp.yP, adp.y, adp.dydt, adp.dydtP);
      }

      doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles, adp.xP,
                     adp.yP, adp.zP, lbb, tfr, wrapMultipliers.getRowPtr(3),
                     wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(4),
                     wrapMultipliers.getRowPtr(1), wrapMultipliers.getRowPtr(5),
                     wrapMultipliers.getRowPtr(2));

      // Error
      error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max, adp.error,
                                                             numBubbles);

      if (error < properties.getErrorTolerance() && timeStep < 0.1)
        timeStep *= 1.9;
      else if (error > properties.getErrorTolerance())
        timeStep *= 0.5;

    } while (error > properties.getErrorTolerance());

    // Update the current values with the calculated predictions
    const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
    CUDA_CALL(cudaMemcpyAsync(adp.dxdtO, adp.dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(adp.x, adp.xP, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(adp.dxdt, adp.dxdtP, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));

    properties.setTimeStep(timeStep);
    elapsedTime += timeStep;

    if (i % 200 == 0)
      updateCellsAndNeighbors();
  }

  // Energy after stabilization
  energy1 = pinnedDouble.get()[1];

  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dummy4);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                  pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x, interval.y,
                  PBC_Y == 1, adp.y, interval.z, PBC_Z == 1, adp.z);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles, pairs.getRowPtr(0),
                  pairs.getRowPtr(1), adp.r, adp.dummy4, interval.x, PBC_X == 1, adp.x, interval.y,
                  PBC_Y == 1, adp.y);

  energy2 =
    cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, numBubbles);

  return elapsedTime;
}

bool Simulator::integrate()
{
  NVTX_RANGE_PUSH_A("Integration function");
  KernelSize kernelSize(128, numBubbles);

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
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtP, adp.dydtP, adp.dzdtP,
                    adp.drdtP, adp.error, adp.dummy1, adp.dummy2);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.xP, adp.x, adp.dxdt,
                    adp.dxdtO, adp.yP, adp.y, adp.dydt, adp.dydtO, adp.zP, adp.z, adp.dzdt,
                    adp.dzdtO, adp.rP, adp.r, adp.drdt, adp.drdtO);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, velocityStream,
                    properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP,
                    interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP, interval.y, lbb.y, PBC_Y == 1,
                    adp.yP, adp.dydtP, interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP);
      // Wall velocity
      doWallVelocity(pairKernelSize, 0, velocityStream, PBC_X == 0, PBC_Y == 0, PBC_Z == 0,
                     numBubbles, pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.xP, adp.yP,
                     adp.zP, adp.dxdtP, adp.dydtP, adp.dzdtP, lbb, tfr);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        int *numNeighbors = bubbleCellIndices.getRowPtr(0);

        KERNEL_LAUNCH(neighborVelocityKernel, pairKernelSize, 0, velocityStream, pairs.getRowPtr(0),
                      pairs.getRowPtr(1), numNeighbors, adp.dummy1, adp.dxdtO, adp.dummy2,
                      adp.dydtO, adp.dummy3, adp.dzdtO);

        KERNEL_LAUNCH(flowVelocityKernel, pairKernelSize, 0, velocityStream, numBubbles,
                      numNeighbors, adp.dxdtP, adp.dydtP, adp.dzdtP, adp.dummy1, adp.dummy2,
                      adp.dummy3, adp.xP, adp.yP, adp.zP, properties.getFlowVel(),
                      properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, velocityStream, numBubbles, timeStep, adp.error,
                    adp.xP, adp.x, adp.dxdt, adp.dxdtP, adp.yP, adp.y, adp.dydt, adp.dydtP, adp.zP,
                    adp.z, adp.dzdt, adp.dzdtP);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, velocityStream, numBubbles, adp.dummy4,
                    adp.s, adp.d, adp.xP, adp.x, adp.x0, wrapMultipliers.getRowPtr(0), interval.x,
                    adp.yP, adp.y, adp.y0, wrapMultipliers.getRowPtr(1), interval.y, adp.zP, adp.z,
                    adp.z0, wrapMultipliers.getRowPtr(2), interval.z);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, velocityStream, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles,
                     adp.xP, adp.yP, adp.zP, lbb, tfr, wrapMultipliers.getRowPtr(3),
                     wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(4),
                     wrapMultipliers.getRowPtr(1), wrapMultipliers.getRowPtr(5),
                     wrapMultipliers.getRowPtr(2));

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, pairKernelSize, 0, gasExchangeStream, numBubbles,
                    pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.drdtP, adp.dummy1,
                    interval.x, PBC_X == 1, adp.xP, interval.y, PBC_Y == 1, adp.yP, interval.z,
                    PBC_Z == 1, adp.zP);
    }
    else // Two dimensions
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, adp.dxdtP, adp.dydtP, adp.drdtP,
                    adp.error, adp.dummy1, adp.dummy2);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, adp.xP, adp.x, adp.dxdt,
                    adp.dxdtO, adp.yP, adp.y, adp.dydt, adp.dydtO, adp.rP, adp.r, adp.drdt,
                    adp.drdtO);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, velocityStream,
                    properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP,
                    interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP, interval.y, lbb.y, PBC_Y == 1,
                    adp.yP, adp.dydtP);
      // Wall velocity
      doWallVelocity(pairKernelSize, 0, velocityStream, PBC_X == 0, PBC_Y == 0, false, numBubbles,
                     pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.xP, adp.yP, adp.zP,
                     adp.dxdtP, adp.dydtP, adp.dzdtP, lbb, tfr);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        int *numNeighbors = bubbleCellIndices.getRowPtr(0);

        KERNEL_LAUNCH(neighborVelocityKernel, pairKernelSize, 0, velocityStream, pairs.getRowPtr(0),
                      pairs.getRowPtr(1), numNeighbors, adp.dummy1, adp.dxdtO, adp.dummy2,
                      adp.dydtO);

        KERNEL_LAUNCH(flowVelocityKernel, pairKernelSize, 0, velocityStream, numBubbles,
                      numNeighbors, adp.dxdtP, adp.dydtP, adp.dzdtP, adp.dummy1, adp.dummy2,
                      adp.dummy3, adp.xP, adp.yP, adp.zP, properties.getFlowVel(),
                      properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, velocityStream, numBubbles, timeStep, adp.error,
                    adp.xP, adp.x, adp.dxdt, adp.dxdtP, adp.yP, adp.y, adp.dydt, adp.dydtP);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, velocityStream, numBubbles, adp.dummy4,
                    adp.s, adp.d, adp.xP, adp.x, adp.x0, wrapMultipliers.getRowPtr(0), interval.x,
                    adp.yP, adp.y, adp.y0, wrapMultipliers.getRowPtr(1), interval.y);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, velocityStream, PBC_X == 1, PBC_Y == 1, false, numBubbles,
                     adp.xP, adp.yP, adp.zP, lbb, tfr, wrapMultipliers.getRowPtr(3),
                     wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(4),
                     wrapMultipliers.getRowPtr(1), wrapMultipliers.getRowPtr(5),
                     wrapMultipliers.getRowPtr(2));

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, pairKernelSize, 0, gasExchangeStream, numBubbles,
                    pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP, adp.drdtP, adp.dummy1,
                    interval.x, PBC_X == 1, adp.xP, interval.y, PBC_Y == 1, adp.yP);
    }

    // Free area
    KERNEL_LAUNCH(freeAreaKernel, kernelSize, 0, gasExchangeStream, numBubbles, adp.rP, adp.dummy1,
                  adp.dummy2, adp.dummy3);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtfa,
                                                         numBubbles, gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy2,
                                                         dtfapr, numBubbles, gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy3, dta,
                                                         numBubbles, gasExchangeStream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, kernelSize, 0, gasExchangeStream, adp.drdtP, adp.rP,
                  adp.dummy1, numBubbles, properties.getKappa(), properties.getKParameter());

    // Radius correct
    KERNEL_LAUNCH(correctKernel, kernelSize, 0, gasExchangeStream, numBubbles, timeStep, adp.error,
                  adp.rP, adp.r, adp.drdt, adp.drdtP);

    // Calculate how many bubbles are below the minimum size.
    // Also take note of maximum radius.
    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, kernelSize, 0, gasExchangeStream, numBubbles,
                  aboveMinRadFlags.getRowPtr(0), adp.rP, properties.getMinRad());

    cubWrapper->reduceNoCopy<int, int *, int *>(
      &cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles,
      gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(
      &cub::DeviceReduce::Max, adp.rP, static_cast<double *>(dtfa), numBubbles, gasExchangeStream);

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int),
                              cudaMemcpyDeviceToHost, gasExchangeStream));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double),
                              cudaMemcpyDeviceToHost, gasExchangeStream));

    // Error
    error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max, adp.error,
                                                           numBubbles);

    if (error < properties.getErrorTolerance() && timeStep < 0.1)
      timeStep *= 1.9;
    else if (error > properties.getErrorTolerance())
      timeStep *= 0.5;

    ++numLoopsDone;

    NVTX_RANGE_POP();
  } while (error > properties.getErrorTolerance());

  // Update values
  const size_t numBytesToCopy = 4 * sizeof(double) * dataStride;

  CUDA_CALL(cudaMemcpyAsync(adp.dxdtO, adp.dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(adp.x, adp.xP, 2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
  CUDA_CALL(
    cudaMemcpyAsync(adp.s, adp.dummy4, sizeof(double) * dataStride, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(3),
                            wrapMultipliers.getSizeInBytes() / 2, cudaMemcpyDeviceToDevice));

  ++integrationStep;
  properties.setTimeStep(timeStep);
  simulationTime += timeStep;

  maxBubbleRadius = pinnedDouble.get()[0];

  // Delete & reorder
  const int numBubblesAboveMinRad = pinnedInt.get()[0];
  const bool shouldDeleteBubbles  = numBubblesAboveMinRad < numBubbles;

  if (shouldDeleteBubbles)
    deleteSmallBubbles(numBubblesAboveMinRad);

  if (shouldDeleteBubbles || integrationStep % 200 == 0)
    updateCellsAndNeighbors();

  bool continueSimulation = numBubbles > properties.getMinNumBubbles();
  continueSimulation &=
    (NUM_DIM == 3) ? maxBubbleRadius < 0.5 * (tfr - lbb).getMinComponent() : true;

  NVTX_RANGE_POP();

  return continueSimulation;
}

void Simulator::updateCellsAndNeighbors()
{
  int *offsets  = cellData.getRowPtr((size_t)CellProperty::OFFSET);
  int *sizes    = cellData.getRowPtr((size_t)CellProperty::SIZE);
  dim3 gridSize = getGridSize();
  const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

  cellData.setBytesToZero();
  bubbleCellIndices.setBytesToZero();

  KernelSize kernelSize(128, numBubbles);

  KERNEL_LAUNCH(assignBubblesToCells, pairKernelSize, 0, 0, adp.x, adp.y, adp.z,
                bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3), properties.getLbb(),
                properties.getTfr(), cellDim, numBubbles);

  int *cellIndices   = bubbleCellIndices.getRowPtr(0);
  int *bubbleIndices = bubbleCellIndices.getRowPtr(1);

  cubWrapper->sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(bubbleCellIndices.getRowPtr(2)),
    cellIndices, const_cast<const int *>(bubbleCellIndices.getRowPtr(3)), bubbleIndices,
    numBubbles);

  CUDA_CALL(cudaEventRecord(blockingEvent1));

  cubWrapper->histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven,
                                              bubbleCellIndices.getRowPtr(2), sizes,
                                              maxNumCells + 1, 0, maxNumCells, numBubbles);

  cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, maxNumCells);
  CUDA_CALL(cudaEventRecord(blockingEvent2));

  CUDA_CALL(cudaStreamWaitEvent(nonBlockingStream, blockingEvent1, 0));
  KERNEL_LAUNCH(
    reorganizeKernel, kernelSize, 0, nonBlockingStream, numBubbles, ReorganizeType::COPY_FROM_INDEX,
    bubbleIndices, bubbleIndices, adp.x, adp.xP, adp.y, adp.yP, adp.z, adp.zP, adp.r, adp.rP,
    adp.dxdt, adp.dxdtP, adp.dydt, adp.dydtP, adp.dzdt, adp.dzdtP, adp.drdt, adp.drdtP, adp.dxdtO,
    adp.error, adp.dydtO, adp.dummy1, adp.dzdtO, adp.dummy2, adp.drdtO, adp.dummy3, adp.x0,
    adp.dummy4, adp.y0, adp.dummy5, adp.z0, adp.dummy6, adp.s, adp.dummy7, adp.d, adp.dummy8,
    wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(3), wrapMultipliers.getRowPtr(1),
    wrapMultipliers.getRowPtr(4), wrapMultipliers.getRowPtr(2), wrapMultipliers.getRowPtr(5));

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x), static_cast<void *>(adp.xP),
                            sizeof(double) * numAliases / 2 * dataStride, cudaMemcpyDeviceToDevice,
                            nonBlockingStream));

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrapMultipliers.getRowPtr(0)),
                            static_cast<void *>(wrapMultipliers.getRowPtr(3)),
                            wrapMultipliers.getSizeInBytes() / 2, cudaMemcpyDeviceToDevice,
                            nonBlockingStream));

  CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream));

  dvec interval = properties.getTfr() - properties.getLbb();

  kernelSize.block = dim3(128, 1, 1);
  kernelSize.grid  = gridSize;

  CUDA_CALL(cudaMemset(np, 0, sizeof(int)));

  CUDA_CALL(cudaStreamWaitEvent(gasExchangeStream, blockingEvent1, 0));
  CUDA_CALL(cudaStreamWaitEvent(gasExchangeStream, blockingEvent2, 0));
  CUDA_CALL(cudaStreamWaitEvent(velocityStream, blockingEvent1, 0));
  CUDA_CALL(cudaStreamWaitEvent(velocityStream, blockingEvent2, 0));

  for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
  {
    cudaStream_t stream = (i % 2) ? velocityStream : gasExchangeStream;
    if (NUM_DIM == 3)
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, numBubbles, maxNumCells,
                    static_cast<int>(pairs.getWidth()), offsets, sizes, pairs.getRowPtr(2),
                    pairs.getRowPtr(3), adp.r, interval.x, PBC_X == 1, adp.x, interval.y,
                    PBC_Y == 1, adp.y, interval.z, PBC_Z == 1, adp.z);
    else
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, numBubbles, maxNumCells,
                    static_cast<int>(pairs.getWidth()), offsets, sizes, pairs.getRowPtr(2),
                    pairs.getRowPtr(3), adp.r, interval.x, PBC_X == 1, adp.x, interval.y,
                    PBC_Y == 1, adp.y);
  }

  CUDA_CALL(
    cudaMemcpy(static_cast<void *>(pinnedInt.get()), np, sizeof(int), cudaMemcpyDeviceToHost));
  int numPairs = pinnedInt.get()[0];
  cubWrapper->sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(pairs.getRowPtr(2)),
    pairs.getRowPtr(0), const_cast<const int *>(pairs.getRowPtr(3)), pairs.getRowPtr(1), numPairs);
}

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
  NVTX_RANGE_PUSH_A("BubbleRemoval");
  KernelSize kernelSize(128, numBubbles);

  int *flag = aboveMinRadFlags.getRowPtr(0);

  CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(double)));
  KERNEL_LAUNCH(calculateRedistributedGasVolume, kernelSize, 0, 0, adp.dummy1, adp.r, flag,
                numBubbles);

  cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtv,
                                                       numBubbles);

  int *newIdx = aboveMinRadFlags.getRowPtr(1);
  cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

  KERNEL_LAUNCH(
    reorganizeKernel, kernelSize, 0, 0, numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx,
    flag, adp.x, adp.xP, adp.y, adp.yP, adp.z, adp.zP, adp.r, adp.rP, adp.dxdt, adp.dxdtP, adp.dydt,
    adp.dydtP, adp.dzdt, adp.dzdtP, adp.drdt, adp.drdtP, adp.dxdtO, adp.error, adp.dydtO,
    adp.dummy1, adp.dzdtO, adp.dummy2, adp.drdtO, adp.dummy3, adp.x0, adp.dummy4, adp.y0,
    adp.dummy5, adp.z0, adp.dummy6, adp.s, adp.dummy7, adp.d, adp.dummy8,
    wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(3), wrapMultipliers.getRowPtr(1),
    wrapMultipliers.getRowPtr(4), wrapMultipliers.getRowPtr(2), wrapMultipliers.getRowPtr(5));

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x), static_cast<void *>(adp.xP),
                            sizeof(double) * numAliases / 2 * dataStride,
                            cudaMemcpyDeviceToDevice));

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrapMultipliers.getRowPtr(0)),
                            static_cast<void *>(wrapMultipliers.getRowPtr(3)),
                            wrapMultipliers.getSizeInBytes() / 2, cudaMemcpyDeviceToDevice));

  numBubbles = numBubblesAboveMinRad;
  KERNEL_LAUNCH(addVolume, kernelSize, 0, 0, adp.r, numBubbles);

  NVTX_RANGE_POP();
}

dim3 Simulator::getGridSize()
{
  const int totalNumCells = std::ceil((float)numBubbles / properties.getNumBubblesPerCell());
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

void Simulator::transformPositions(bool normalize)
{
  KERNEL_LAUNCH(transformPositionsKernel, pairKernelSize, 0, 0, normalize, numBubbles,
                properties.getLbb(), properties.getTfr(), adp.x, adp.y, adp.z);
}

double Simulator::getAverageProperty(double *p)
{
  return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, p, numBubbles) /
         numBubbles;
}

void Simulator::saveSnapshotToFile()
{
  std::stringstream ss;
  ss << properties.getSnapshotFilename() << ".csv." << numSnapshots;
  std::ofstream file(ss.str().c_str(), std::ios::out);
  if (file.is_open())
  {
    const size_t numComp = 17;
    hostData.clear();
    hostData.resize(dataStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), deviceData, sizeof(double) * numComp * dataStride,
                         cudaMemcpyDeviceToHost));

    file << "x,y,z,r,vx,vy,vz,path,dist\n";
    for (size_t i = 0; i < (size_t)numBubbles; ++i)
    {
      file << hostData[i + 0 * dataStride];
      file << ",";
      file << hostData[i + 1 * dataStride];
      file << ",";
      file << hostData[i + 2 * dataStride];
      file << ",";
      file << hostData[i + 3 * dataStride];
      file << ",";
      file << hostData[i + 4 * dataStride];
      file << ",";
      file << hostData[i + 5 * dataStride];
      file << ",";
      file << hostData[i + 6 * dataStride];
      file << ",";
      file << hostData[i + 15 * dataStride];
      file << ",";
      file << hostData[i + 16 * dataStride];
      file << "\n";
    }

    ++numSnapshots;
  }
}

void Simulator::startProfiling(bool start)
{
  if (start)
    cudaProfilerStart();
}

void Simulator::stopProfiling(bool stop, bool &continueIntegration)
{
  if (stop)
  {
    cudaProfilerStop();
    continueIntegration = false;
  }
}

void Simulator::doBoundaryWrap(KernelSize ks, int sm, cudaStream_t stream, bool wrapX, bool wrapY,
                               bool wrapZ, int numValues, double *x, double *y, double *z, dvec lbb,
                               dvec tfr, int *mulX, int *mulY, int *mulZ, int *mulOldX,
                               int *mulOldY, int *mulOldZ)
{
  if (wrapX && wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y,
                  lbb.y, tfr.y, mulY, mulOldY, z, lbb.z, tfr.z, mulZ, mulOldZ);
  else if (wrapX && wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y,
                  lbb.y, tfr.y, mulY, mulOldY);
  else if (wrapX && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, z,
                  lbb.z, tfr.z, mulZ, mulOldZ);
  else if (wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY, z,
                  lbb.z, tfr.z, mulZ, mulOldZ);
  else if (wrapX)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX);
  else if (wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY);
  else if (wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, z, lbb.z, tfr.z, mulZ, mulOldZ);
}

void Simulator::doWallVelocity(KernelSize ks, int sm, cudaStream_t stream, bool doX, bool doY,
                               bool doZ, int numValues, int *first, int *second, double *r,
                               double *x, double *y, double *z, double *dxdt, double *dydt,
                               double *dzdt, dvec lbb, dvec tfr)
{
  dvec interval = tfr - lbb;
  if (doX && doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y,
                  dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX && doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y,
                  dydt);
  }
  else if (doX && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.x, lbb.x, !doX, x, dxdt, interval.z, lbb.z, !doZ, z,
                  dzdt);
  }
  else if (doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z,
                  dzdt);
  }
  else if (doX)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.x, lbb.x, !doX, x, dxdt);
  }
  else if (doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, properties.getFZeroPerMuZero(),
                  first, second, r, interval.z, lbb.z, !doZ, z, dzdt);
  }
}
} // namespace cubble
