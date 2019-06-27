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

    KERNEL_LAUNCH(calculateVolumes, kernelSize, 0, 0, ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::TEMP1], numBubbles);

    return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                          ddps[(uint32_t)DDP::TEMP1], numBubbles);
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
    // Set starting positions and reset wrap counts to 0
    const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
    CUDA_CALL(cudaMemcpy(ddps[(uint32_t)DDP::X0], ddps[(uint32_t)DDP::X], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemset(dips[(uint32_t)DIP::WRAP_COUNT_X], 0, 6 * dataStride * sizeof(int)));

    simulationTime      = 0;
    int timesPrinted    = 1;
    uint32_t numSteps   = 0;
    const dvec interval = properties.getTfr() - properties.getLbb();

    KernelSize kernelSize(128, numBubbles);

    // Calculate the energy at simulation start
    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::TEMP4]);

    if (NUM_DIM == 3)
    {
      KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                    dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                    ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                    interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1,
                    ddps[(uint32_t)DDP::Z]);
    }
    else
    {
      KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                    dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                    ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                    interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y]);
    }

    energy1 = cubWrapper->reduce<double, double *, double *>(
      &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP4], numBubbles);

    // Start the simulation proper
    bool continueIntegration = true;
    int numTotalSteps        = 0;
    std::cout << "T\tR\t#b\tdE\t\t#steps\t#pairs/#b" << std::endl;
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
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::TEMP4]);

        if (NUM_DIM == 3)
          KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                        dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                        ddps[(uint32_t)DDP::R], ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                        ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y],
                        interval.z, PBC_Z == 1, ddps[(uint32_t)DDP::Z]);
        else
          KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                        dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                        ddps[(uint32_t)DDP::R], ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                        ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y]);

        energy2 = cubWrapper->reduce<double, double *, double *>(
          &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP4], numBubbles);
        const double dE = (energy2 - energy1) / energy2;

        // Add values to data stream
        double relativeRadius = getAverageProperty(ddps[(uint32_t)DDP::R]) / properties.getAvgRad();
        dataStream << scaledTime << " " << relativeRadius << " "
                   << maxBubbleRadius / properties.getAvgRad() << " " << numBubbles << " "
                   << getAverageProperty(ddps[(uint32_t)DDP::PATH]) << " "
                   << getAverageProperty(ddps[(uint32_t)DDP::DISTANCE]) << " " << dE << "\n";

        // Print some values
        std::cout << scaledTime << "\t" << relativeRadius << "\t" << numBubbles << "\t" << dE
                  << "\t" << numSteps << "\t" << (float)numPairs / numBubbles << std::endl;

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
  // First calculate the size of the box and the starting number of bubbles
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

  // Initialize CUB
  cubWrapper = std::make_shared<CubWrapper>(numBubbles * sizeof(double));

  // Determine the maximum number of Morton numbers for the simulation box
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

  std::cout << "Maximum (theoretical) number of cells: " << maxNumCells
            << ", actual grid dimensions: " << gridDim.x << ", " << gridDim.y << ", " << gridDim.z
            << std::endl;

  // Reserve memory for data
  reserveMemory();
  std::cout << "Memory requirement for data:\n\tdouble: " << memReqD << " bytes\n\tint: " << memReqI
            << " bytes" << std::endl;

  // Get some device global symbol addresses to host pointers.
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfa), dTotalFreeArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfapr), dTotalFreeAreaPerRadius));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&mbpc), dMaxBubblesPerCell));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dvm), dVolumeMultiplier));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtv), dTotalVolume));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&np), dNumPairs));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dir), dInvRho));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dta), dTotalArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dasai), dAverageSurfaceAreaIn));

  // Streams
  CUDA_ASSERT(cudaStreamCreate(&velocityStream));
  CUDA_ASSERT(cudaStreamCreate(&gasExchangeStream));

  printRelevantInfoOfCurrentDevice();

  pairKernelSize.block = dim3(128, 1, 1);
  pairKernelSize.grid  = dim3(256, 1, 1);
  KernelSize kernelSize(128, numBubbles);

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
    CURAND_CALL(curandGenerateUniformDouble(generator, ddps[(uint32_t)DDP::Z], numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, ddps[(uint32_t)DDP::X], numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, ddps[(uint32_t)DDP::Y], numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, ddps[(uint32_t)DDP::RP], numBubbles));
  CURAND_CALL(
    curandGenerateNormalDouble(generator, ddps[(uint32_t)DDP::R], numBubbles, avgRad, stdDevRad));
  CURAND_CALL(curandDestroyGenerator(generator));

  KERNEL_LAUNCH(assignDataToBubbles, kernelSize, 0, 0, ddps[(uint32_t)DDP::X],
                ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::XP],
                ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::R],
                ddps[(uint32_t)DDP::RP], dips[(uint32_t)DIP::FLAGS], bubblesPerDim, tfr, lbb,
                avgRad, properties.getMinRad(), numBubbles);

  cubWrapper->reduceNoCopy<double, double *, double *>(
    &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::RP], dasai, numBubbles, 0);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(ddps[(uint32_t)DDP::RP]),
                            static_cast<void *>(ddps[(uint32_t)DDP::R]),
                            sizeof(double) * dataStride, cudaMemcpyDeviceToDevice, 0));

  // Delete small bubbles, if any
  const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(
    &cub::DeviceReduce::Sum, dips[(uint32_t)DIP::FLAGS], numBubbles);
  if (numBubblesAboveMinRad < numBubbles)
    deleteSmallBubbles(numBubblesAboveMinRad);

  maxBubbleRadius = cubWrapper->reduce<double, double *, double *>(
    &cub::DeviceReduce::Max, ddps[(uint32_t)DDP::R], numBubbles);

  updateCellsAndNeighbors();

  // Calculate some initial values which are needed
  // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTO],
                ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::DZDTO], ddps[(uint32_t)DDP::DRDTO],
                ddps[(uint32_t)DDP::DISTANCE], ddps[(uint32_t)DDP::PATH]);

  std::cout << "Calculating some initial values as a part of setup." << std::endl;

  if (NUM_DIM == 3)
  {
    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDTO],
                  interval.z, lbb.z, PBC_Z == 1, ddps[(uint32_t)DDP::Z],
                  ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, numBubbles, timeStep, ddps[(uint32_t)DDP::X],
                  ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDTO],
                  ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::DZDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles,
                   ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], lbb,
                   tfr, dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                   dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                   dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTO],
                  ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::DZDTO],
                  ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDTO],
                  interval.z, lbb.z, PBC_Z == 1, ddps[(uint32_t)DDP::Z],
                  ddps[(uint32_t)DDP::DZDTO]);
  }
  else
  {
    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y],
                  ddps[(uint32_t)DDP::DYDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, numBubbles, timeStep, ddps[(uint32_t)DDP::X],
                  ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, false, numBubbles,
                   ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], lbb,
                   tfr, dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                   dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                   dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTO],
                  ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  interval.x, lbb.x, PBC_X == 1, ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDTO],
                  interval.y, lbb.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y],
                  ddps[(uint32_t)DDP::DYDTO]);
  }
}

void Simulator::deinit()
{
  saveSnapshotToFile();
  properties.writeParameters();

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(static_cast<void *>(deviceDoubles)));
  CUDA_CALL(cudaFree(static_cast<void *>(deviceInts)));

  CUDA_CALL(cudaStreamDestroy(velocityStream));
  CUDA_CALL(cudaStreamDestroy(gasExchangeStream));
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
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                  interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1,
                  ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                  interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y]);

  cubWrapper->reduceNoCopy<double, double *, double *>(
    &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP4], dtfapr, numBubbles);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&pinnedDouble.get()[1]),
                            static_cast<void *>(dtfapr), sizeof(double), cudaMemcpyDeviceToHost,
                            0));

  for (int i = 0; i < properties.getNumStepsToRelax(); ++i)
  {
    do
    {
      if (NUM_DIM == 3)
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTP],
                      ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP],
                      ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::TEMP1],
                      ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep,
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDT],
                      ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y],
                      ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTO],
                      ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::DZDT],
                      ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                      ddps[(uint32_t)DDP::RP], interval.x, lbb.x, PBC_X == 1,
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::DXDTP], interval.y, lbb.y,
                      PBC_Y == 1, ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::DYDTP], interval.z,
                      lbb.z, PBC_Z == 1, ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DZDTP]);

        doWallVelocity(pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, numBubbles,
                       dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                       ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP],
                       ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DXDTP],
                       ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP], lbb, tfr);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, numBubbles, timeStep,
                      ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                      ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP],
                      ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT],
                      ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::Z],
                      ddps[(uint32_t)DDP::DZDT], ddps[(uint32_t)DDP::DZDTP]);
      }
      else // Two dimensional case
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTP],
                      ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::ERROR],
                      ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep,
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDT],
                      ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y],
                      ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0, properties.getFZeroPerMuZero(),
                      dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                      ddps[(uint32_t)DDP::RP], interval.x, lbb.x, PBC_X == 1,
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::DXDTP], interval.y, lbb.y,
                      PBC_Y == 1, ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::DYDTP]);

        doWallVelocity(pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, numBubbles,
                       dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                       ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP],
                       ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DXDTP],
                       ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP], lbb, tfr);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, numBubbles, timeStep,
                      ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                      ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP],
                      ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT],
                      ddps[(uint32_t)DDP::DYDTP]);
      }

      doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles,
                     ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Error
      error = cubWrapper->reduce<double, double *, double *>(
        &cub::DeviceReduce::Max, ddps[(uint32_t)DDP::ERROR], numBubbles);

      if (error < properties.getErrorTolerance() && timeStep < 0.1)
        timeStep *= 1.9;
      else if (error > properties.getErrorTolerance())
        timeStep *= 0.5;

    } while (error > properties.getErrorTolerance());

    // Update the current values with the calculated predictions
    const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
    CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::XP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));

    properties.setTimeStep(timeStep);
    elapsedTime += timeStep;

    if (i % 200 == 0)
      updateCellsAndNeighbors();
  }

  // Energy after stabilization
  energy1 = pinnedDouble.get()[1];

  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                  interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1,
                  ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0, numBubbles,
                  dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1, ddps[(uint32_t)DDP::X],
                  interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y]);

  energy2 = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                           ddps[(uint32_t)DDP::TEMP4], numBubbles);

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
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTP],
                    ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP],
                    ddps[(uint32_t)DDP::DRDTP], ddps[(uint32_t)DDP::ERROR],
                    ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, ddps[(uint32_t)DDP::XP],
                    ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTO],
                    ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT],
                    ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::Z],
                    ddps[(uint32_t)DDP::DZDT], ddps[(uint32_t)DDP::DZDTO], ddps[(uint32_t)DDP::RP],
                    ddps[(uint32_t)DDP::R], ddps[(uint32_t)DDP::DRDT], ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(
        velocityPairKernel, pairKernelSize, 0, velocityStream, properties.getFZeroPerMuZero(),
        dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::RP], interval.x,
        lbb.x, PBC_X == 1, ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::DXDTP], interval.y, lbb.y,
        PBC_Y == 1, ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::DYDTP], interval.z, lbb.z,
        PBC_Z == 1, ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DZDTP]);
      // Wall velocity
      doWallVelocity(pairKernelSize, 0, velocityStream, PBC_X == 0, PBC_Y == 0, PBC_Z == 0,
                     numBubbles, dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                     ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP],
                     ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DXDTP],
                     ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP], lbb, tfr);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * pairStride));
        int *numNeighbors = dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, pairKernelSize, 0, velocityStream,
                      dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::DXDTO],
                      ddps[(uint32_t)DDP::TEMP2], ddps[(uint32_t)DDP::DYDTO],
                      ddps[(uint32_t)DDP::TEMP3], ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, pairKernelSize, 0, velocityStream, numBubbles,
                      numNeighbors, ddps[(uint32_t)DDP::DXDTP], ddps[(uint32_t)DDP::DYDTP],
                      ddps[(uint32_t)DDP::DZDTP], ddps[(uint32_t)DDP::TEMP1],
                      ddps[(uint32_t)DDP::TEMP2], ddps[(uint32_t)DDP::TEMP3],
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP],
                      properties.getFlowVel(), properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, velocityStream, numBubbles, timeStep,
                    ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                    ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP], ddps[(uint32_t)DDP::YP],
                    ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTP],
                    ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::DZDT],
                    ddps[(uint32_t)DDP::DZDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, velocityStream, numBubbles,
                    ddps[(uint32_t)DDP::TEMP4], ddps[(uint32_t)DDP::PATH],
                    ddps[(uint32_t)DDP::DISTANCE], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                    ddps[(uint32_t)DDP::X0], dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x,
                    ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::Y0],
                    dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y, ddps[(uint32_t)DDP::ZP],
                    ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::Z0],
                    dips[(uint32_t)DIP::WRAP_COUNT_ZP], interval.z);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, velocityStream, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, numBubbles,
                     ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, pairKernelSize, 0, gasExchangeStream, numBubbles,
                    dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::RP],
                    ddps[(uint32_t)DDP::DRDTP], ddps[(uint32_t)DDP::TEMP1], interval.x, PBC_X == 1,
                    ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::YP],
                    interval.z, PBC_Z == 1, ddps[(uint32_t)DDP::ZP]);
    }
    else // Two dimensions
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, numBubbles, ddps[(uint32_t)DDP::DXDTP],
                    ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DRDTP],
                    ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::TEMP1],
                    ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, numBubbles, timeStep, ddps[(uint32_t)DDP::XP],
                    ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTO],
                    ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT],
                    ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::R],
                    ddps[(uint32_t)DDP::DRDT], ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, velocityStream,
                    properties.getFZeroPerMuZero(), dips[(uint32_t)DIP::PAIR1],
                    dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::RP], interval.x, lbb.x,
                    PBC_X == 1, ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::DXDTP], interval.y,
                    lbb.y, PBC_Y == 1, ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::DYDTP]);
      // Wall velocity
      doWallVelocity(pairKernelSize, 0, velocityStream, PBC_X == 0, PBC_Y == 0, false, numBubbles,
                     dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2],
                     ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP],
                     ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::DXDTP],
                     ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDTP], lbb, tfr);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * pairStride));
        int *numNeighbors = dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, pairKernelSize, 0, velocityStream,
                      dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::DXDTO],
                      ddps[(uint32_t)DDP::TEMP2], ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, pairKernelSize, 0, velocityStream, numBubbles,
                      numNeighbors, ddps[(uint32_t)DDP::DXDTP], ddps[(uint32_t)DDP::DYDTP],
                      ddps[(uint32_t)DDP::DZDTP], ddps[(uint32_t)DDP::TEMP1],
                      ddps[(uint32_t)DDP::TEMP2], ddps[(uint32_t)DDP::TEMP3],
                      ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP],
                      properties.getFlowVel(), properties.getFlowTfr(), properties.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, velocityStream, numBubbles, timeStep,
                    ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                    ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP], ddps[(uint32_t)DDP::YP],
                    ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, velocityStream, numBubbles,
                    ddps[(uint32_t)DDP::TEMP4], ddps[(uint32_t)DDP::PATH],
                    ddps[(uint32_t)DDP::DISTANCE], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::X],
                    ddps[(uint32_t)DDP::X0], dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x,
                    ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::Y0],
                    dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, velocityStream, PBC_X == 1, PBC_Y == 1, false, numBubbles,
                     ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::ZP], lbb,
                     tfr, dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, pairKernelSize, 0, gasExchangeStream, numBubbles,
                    dips[(uint32_t)DIP::PAIR1], dips[(uint32_t)DIP::PAIR2], ddps[(uint32_t)DDP::RP],
                    ddps[(uint32_t)DDP::DRDTP], ddps[(uint32_t)DDP::TEMP1], interval.x, PBC_X == 1,
                    ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::YP]);
    }

    // Free area
    KERNEL_LAUNCH(freeAreaKernel, kernelSize, 0, gasExchangeStream, numBubbles,
                  ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::TEMP2],
                  ddps[(uint32_t)DDP::TEMP3]);

    cubWrapper->reduceNoCopy<double, double *, double *>(
      &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP1], dtfa, numBubbles, gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(
      &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP2], dtfapr, numBubbles, gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(
      &cub::DeviceReduce::Sum, ddps[(uint32_t)DDP::TEMP3], dta, numBubbles, gasExchangeStream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, kernelSize, 0, gasExchangeStream,
                  ddps[(uint32_t)DDP::DRDTP], ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::TEMP1],
                  numBubbles, properties.getKappa(), properties.getKParameter());

    // Radius correct
    KERNEL_LAUNCH(correctKernel, kernelSize, 0, gasExchangeStream, numBubbles, timeStep,
                  ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::R],
                  ddps[(uint32_t)DDP::DRDT], ddps[(uint32_t)DDP::DRDTP]);

    // Calculate how many bubbles are below the minimum size.
    // Also take note of maximum radius.
    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, kernelSize, 0, gasExchangeStream, numBubbles,
                  dips[(uint32_t)DIP::FLAGS], ddps[(uint32_t)DDP::RP], properties.getMinRad());

    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, dips[(uint32_t)DIP::FLAGS],
                                                static_cast<int *>(mbpc), numBubbles,
                                                gasExchangeStream);
    cubWrapper->reduceNoCopy<double, double *, double *>(
      &cub::DeviceReduce::Max, ddps[(uint32_t)DDP::RP], static_cast<double *>(dtfa), numBubbles,
      gasExchangeStream);

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int),
                              cudaMemcpyDeviceToHost, gasExchangeStream));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double),
                              cudaMemcpyDeviceToHost, gasExchangeStream));

    // Error
    error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max,
                                                           ddps[(uint32_t)DDP::ERROR], numBubbles);

    if (error < properties.getErrorTolerance() && timeStep < 0.1)
      timeStep *= 1.9;
    else if (error > properties.getErrorTolerance())
      timeStep *= 0.5;

    ++numLoopsDone;

    NVTX_RANGE_POP();
  } while (error > properties.getErrorTolerance());

  // Update values
  const size_t numBytesToCopy = 4 * sizeof(double) * dataStride;

  CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::DXDT], numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::XP], 2 * numBytesToCopy,
                            cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(ddps[(uint32_t)DDP::PATH], ddps[(uint32_t)DDP::TEMP4],
                            sizeof(double) * dataStride, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(dips[(uint32_t)DIP::WRAP_COUNT_XP], dips[(uint32_t)DIP::WRAP_COUNT_X],
                            dataStride * 3 * sizeof(int), cudaMemcpyDeviceToDevice));

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
  dim3 gridSize = getGridSize();
  const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

  int *offsets             = dips[(uint32_t)DIP::PAIR1];
  int *sizes               = dips[(uint32_t)DIP::PAIR1] + maxNumCells;
  int *cellIndices         = dips[(uint32_t)DIP::TEMP1] + 0 * dataStride;
  int *bubbleIndices       = dips[(uint32_t)DIP::TEMP1] + 1 * dataStride;
  int *sortedCellIndices   = dips[(uint32_t)DIP::TEMP1] + 2 * dataStride;
  int *sortedBubbleIndices = dips[(uint32_t)DIP::TEMP1] + 3 * dataStride;

  const size_t resetBytes =
    sizeof(int) * pairStride * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
  CUDA_CALL(cudaMemset(dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

  KernelSize kernelSize(128, numBubbles);

  KERNEL_LAUNCH(assignBubblesToCells, pairKernelSize, 0, 0, ddps[(uint32_t)DDP::X],
                ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                properties.getLbb(), properties.getTfr(), cellDim, numBubbles);

  cubWrapper->sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices), sortedCellIndices,
    const_cast<const int *>(bubbleIndices), sortedBubbleIndices, numBubbles);

  cubWrapper->histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven, cellIndices,
                                              sizes, maxNumCells + 1, 0, maxNumCells, numBubbles);

  cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, maxNumCells);

  KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, 0, numBubbles, ReorganizeType::COPY_FROM_INDEX,
                sortedBubbleIndices, sortedBubbleIndices, ddps[(uint32_t)DDP::X],
                ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::YP],
                ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::ZP], ddps[(uint32_t)DDP::R],
                ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::DXDT], ddps[(uint32_t)DDP::DXDTP],
                ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTP], ddps[(uint32_t)DDP::DZDT],
                ddps[(uint32_t)DDP::DZDTP], ddps[(uint32_t)DDP::DRDT], ddps[(uint32_t)DDP::DRDTP],
                ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::ERROR], ddps[(uint32_t)DDP::DYDTO],
                ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::DZDTO], ddps[(uint32_t)DDP::TEMP2],
                ddps[(uint32_t)DDP::DRDTO], ddps[(uint32_t)DDP::TEMP3], ddps[(uint32_t)DDP::X0],
                ddps[(uint32_t)DDP::TEMP4], ddps[(uint32_t)DDP::Y0], ddps[(uint32_t)DDP::TEMP5],
                ddps[(uint32_t)DDP::Z0], ddps[(uint32_t)DDP::TEMP6], ddps[(uint32_t)DDP::PATH],
                ddps[(uint32_t)DDP::TEMP7], ddps[(uint32_t)DDP::DISTANCE],
                ddps[(uint32_t)DDP::TEMP8], dips[(uint32_t)DIP::WRAP_COUNT_X],
                dips[(uint32_t)DIP::WRAP_COUNT_XP], dips[(uint32_t)DIP::WRAP_COUNT_Y],
                dips[(uint32_t)DIP::WRAP_COUNT_YP], dips[(uint32_t)DIP::WRAP_COUNT_Z],
                dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(ddps[(uint32_t)DDP::XP]), memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  dvec interval = properties.getTfr() - properties.getLbb();

  kernelSize.block = dim3(128, 1, 1);
  kernelSize.grid  = gridSize;

  CUDA_CALL(cudaMemset(np, 0, sizeof(int)));

  const double maxDistance = 1.5 * maxBubbleRadius;

  for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
  {
    cudaStream_t stream = (i % 2) ? velocityStream : gasExchangeStream;
    if (NUM_DIM == 3)
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, numBubbles, maxNumCells,
                    (int)pairStride, maxDistance, offsets, sizes, dips[(uint32_t)DIP::TEMP1],
                    dips[(uint32_t)DIP::TEMP2], ddps[(uint32_t)DDP::R], interval.x, PBC_X == 1,
                    ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y],
                    interval.z, PBC_Z == 1, ddps[(uint32_t)DDP::Z]);
    else
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, numBubbles, maxNumCells,
                    (int)pairStride, maxDistance, offsets, sizes, dips[(uint32_t)DIP::TEMP1],
                    dips[(uint32_t)DIP::TEMP2], ddps[(uint32_t)DDP::R], interval.x, PBC_X == 1,
                    ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, ddps[(uint32_t)DDP::Y]);
  }

  CUDA_CALL(
    cudaMemcpy(static_cast<void *>(pinnedInt.get()), np, sizeof(int), cudaMemcpyDeviceToHost));
  numPairs = pinnedInt.get()[0];
  cubWrapper->sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(dips[(uint32_t)DIP::TEMP1]),
    dips[(uint32_t)DIP::PAIR1], const_cast<const int *>(dips[(uint32_t)DIP::TEMP2]),
    dips[(uint32_t)DIP::PAIR2], numPairs);
}

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
  NVTX_RANGE_PUSH_A("BubbleRemoval");
  KernelSize kernelSize(128, numBubbles);

  CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(double)));
  KERNEL_LAUNCH(calculateRedistributedGasVolume, kernelSize, 0, 0, ddps[(uint32_t)DDP::TEMP1],
                ddps[(uint32_t)DDP::R], dips[(uint32_t)DIP::FLAGS], numBubbles);

  cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                       ddps[(uint32_t)DDP::TEMP1], dtv, numBubbles);

  int *newIdx = dips[(uint32_t)DIP::TEMP1];
  cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, dips[(uint32_t)DIP::FLAGS], newIdx,
                                 numBubbles);

  KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, 0, numBubbles,
                ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, dips[(uint32_t)DIP::FLAGS],
                ddps[(uint32_t)DDP::X], ddps[(uint32_t)DDP::XP], ddps[(uint32_t)DDP::Y],
                ddps[(uint32_t)DDP::YP], ddps[(uint32_t)DDP::Z], ddps[(uint32_t)DDP::ZP],
                ddps[(uint32_t)DDP::R], ddps[(uint32_t)DDP::RP], ddps[(uint32_t)DDP::DXDT],
                ddps[(uint32_t)DDP::DXDTP], ddps[(uint32_t)DDP::DYDT], ddps[(uint32_t)DDP::DYDTP],
                ddps[(uint32_t)DDP::DZDT], ddps[(uint32_t)DDP::DZDTP], ddps[(uint32_t)DDP::DRDT],
                ddps[(uint32_t)DDP::DRDTP], ddps[(uint32_t)DDP::DXDTO], ddps[(uint32_t)DDP::ERROR],
                ddps[(uint32_t)DDP::DYDTO], ddps[(uint32_t)DDP::TEMP1], ddps[(uint32_t)DDP::DZDTO],
                ddps[(uint32_t)DDP::TEMP2], ddps[(uint32_t)DDP::DRDTO], ddps[(uint32_t)DDP::TEMP3],
                ddps[(uint32_t)DDP::X0], ddps[(uint32_t)DDP::TEMP4], ddps[(uint32_t)DDP::Y0],
                ddps[(uint32_t)DDP::TEMP5], ddps[(uint32_t)DDP::Z0], ddps[(uint32_t)DDP::TEMP6],
                ddps[(uint32_t)DDP::PATH], ddps[(uint32_t)DDP::TEMP7],
                ddps[(uint32_t)DDP::DISTANCE], ddps[(uint32_t)DDP::TEMP8],
                dips[(uint32_t)DIP::WRAP_COUNT_X], dips[(uint32_t)DIP::WRAP_COUNT_XP],
                dips[(uint32_t)DIP::WRAP_COUNT_Y], dips[(uint32_t)DIP::WRAP_COUNT_YP],
                dips[(uint32_t)DIP::WRAP_COUNT_Z], dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(ddps[(uint32_t)DDP::XP]), memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  numBubbles = numBubblesAboveMinRad;
  KERNEL_LAUNCH(addVolume, kernelSize, 0, 0, ddps[(uint32_t)DDP::R], numBubbles);

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
                properties.getLbb(), properties.getTfr(), ddps[(uint32_t)DDP::X],
                ddps[(uint32_t)DDP::Y], ddps[(uint32_t)DDP::Z]);
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
    CUDA_CALL(cudaMemcpy(hostData.data(), deviceDoubles, sizeof(double) * numComp * dataStride,
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

void Simulator::reserveMemory()
{
  // Reserve pinned memory
  pinnedInt    = PinnedHostArray<int>(1);
  pinnedDouble = PinnedHostArray<double>(3);

  // Calculate the length of 'rows'. Will be divisible by 32, as that's the warp size.
  dataStride = numBubbles + !!(numBubbles % 32) * (32 - numBubbles % 32);

  // Doubles
  memReqD = sizeof(double) * (uint64_t)dataStride * (uint64_t)DDP::NUM_VALUES;
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&deviceDoubles), memReqD));

  for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
    ddps[i]       = deviceDoubles + i * dataStride;

  // Integers
  const uint32_t avgNumNeighbors = 8;
  pairStride                     = avgNumNeighbors * dataStride;

  memReqI =
    sizeof(int) * (uint64_t)dataStride *
    ((uint64_t)DIP::PAIR1 + avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&deviceInts), memReqI));

  for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
  {
    dips[i]       = deviceInts + i * dataStride;
    std::cout << "int " << i << std::endl;
  }

  uint32_t j = 0;
  for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
  {
    dips[i]       = dips[(uint32_t)DIP::PAIR1] + avgNumNeighbors * ++j * dataStride;
    std::cout << "int " << i << std::endl;
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
