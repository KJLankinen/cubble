// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
#include "Util.h"
#include "Kernels.cuh"

#include "cub/cub/cub.cuh"

#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <curand.h>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

namespace cubble
{
typedef BubbleProperty BP;
Simulator::Simulator(std::shared_ptr<Env> e)
{
    env = e;

#if (NUM_DIM == 3)
    givenNumBubblesPerDim = std::ceil(std::cbrt((float)env->getNumBubbles()));
    numBubbles = givenNumBubblesPerDim * givenNumBubblesPerDim * givenNumBubblesPerDim;
#else
    givenNumBubblesPerDim = std::ceil(std::sqrt((float)env->getNumBubbles()));
    numBubbles = givenNumBubblesPerDim * givenNumBubblesPerDim;
#endif
    cubWrapper = std::make_shared<CubWrapper>(env, numBubbles);
    const dvec tfr = env->getLbb() + env->getAvgRad() * (double)givenNumBubblesPerDim * 2;
    env->setTfr(tfr);

    bubbleData = FixedSizeDeviceArray<double>(numBubbles, (size_t)BP::NUM_VALUES);
    aboveMinRadFlags = FixedSizeDeviceArray<int>(numBubbles, 2);
    bubbleCellIndices = FixedSizeDeviceArray<int>(numBubbles, 4);

    // TODO: Figure out a more sensible value for this.
    const int maxNumPairs = (CUBBLE_NUM_NEIGHBORS + 1) * env->getNumBubblesPerCell() * numBubbles;
    neighborPairIndices = FixedSizeDeviceArray<int>(maxNumPairs, 4);
    numPairs = FixedSizeDeviceArray<int>(1, 1);

    const dim3 gridSize = getGridSize();
    size_t numCells = gridSize.x * gridSize.y * gridSize.z;
    cellData = FixedSizeDeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);

    hostData.resize(bubbleData.getSize(), 0);

    printRelevantInfoOfCurrentDevice();
}

Simulator::~Simulator() {}

void Simulator::setupSimulation()
{
    generateBubbles();
    deleteSmallBubbles();
    updateCellsAndNeighbors();

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton perdictor-corrector method (ABMpc).

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *y = bubbleData.getRowPtr((size_t)BP::Y);
    double *z = bubbleData.getRowPtr((size_t)BP::Z);
    double *r = bubbleData.getRowPtr((size_t)BP::R);

    double *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BP::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BP::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BP::DRDT);

    double *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BP::DRDT_OLD);

    double *energies = bubbleData.getRowPtr((size_t)BP::ENERGY);
    double *freeArea = bubbleData.getRowPtr((size_t)BP::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy accPolicy(128, hostNumPairs);

    double timeStep = env->getTimeStep();
    resetValues(dxdtOld, dydtOld, dzdtOld, drdtOld);
    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, energies, freeArea,
               firstIndices, secondIndices, numBubbles, hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), tfr - lbb, false, false);

    cudaLaunch(defaultPolicy, eulerIntegration,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, tfr, lbb, timeStep, numBubbles);

    if (deleteSmallBubbles())
        updateCellsAndNeighbors();

    resetValues(dxdtOld, dydtOld, dzdtOld, drdtOld);

    cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, energies, freeArea,
               firstIndices, secondIndices, numBubbles, hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), tfr - lbb, false, false);
}

bool Simulator::integrate(bool useGasExchange, bool calculateEnergy)
{
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy accPolicy(128, hostNumPairs);

    double timeStep = env->getTimeStep();
    double error = 0;

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *y = bubbleData.getRowPtr((size_t)BP::Y);
    double *z = bubbleData.getRowPtr((size_t)BP::Z);
    double *r = bubbleData.getRowPtr((size_t)BP::R);

    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    double *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);

    double *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BP::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BP::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BP::DRDT);

    double *dxdtPrd = bubbleData.getRowPtr((size_t)BP::DXDT_PRD);
    double *dydtPrd = bubbleData.getRowPtr((size_t)BP::DYDT_PRD);
    double *dzdtPrd = bubbleData.getRowPtr((size_t)BP::DZDT_PRD);
    double *drdtPrd = bubbleData.getRowPtr((size_t)BP::DRDT_PRD);

    double *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BP::DRDT_OLD);

    double *energies = bubbleData.getRowPtr((size_t)BP::ENERGY);
    double *errors = bubbleData.getRowPtr((size_t)BP::ERROR);
    double *volumes = bubbleData.getRowPtr((size_t)BP::VOLUME);
    double *freeArea = bubbleData.getRowPtr((size_t)BP::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    size_t numLoopsDone = 0;
    do
    {
        resetValues(dxdtPrd, dydtPrd, dzdtPrd, drdtPrd, freeArea, energies, errors);
        cudaLaunch(defaultPolicy, predict, x, y, z, r, xPrd, yPrd, zPrd, rPrd, dxdt, dydt, dzdt, drdt, dxdtOld, dydtOld, dzdtOld, drdtOld, tfr, lbb, timeStep, numBubbles, useGasExchange);

        cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
                   xPrd, yPrd, zPrd, rPrd, dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
                   energies, freeArea, firstIndices, secondIndices, numBubbles,
                   hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), env->getTfr() - env->getLbb(), calculateEnergy, useGasExchange);

        if (useGasExchange)
        {
            cudaLaunch(defaultPolicy, calculateFreeAreaPerRadius,
                       rPrd, freeArea, errors, env->getPi(), numBubbles);
            double invRho = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, errors, numBubbles);
            invRho /= cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, freeArea, numBubbles);
            cudaLaunch(defaultPolicy, calculateFinalRadiusChangeRate,
                       drdtPrd, rPrd, freeArea, numBubbles, invRho, 1.0 / env->getPi(), env->getKappa(), env->getKParameter());
        }

        cudaLaunch(defaultPolicy, correct,
                   x, y, z, r, xPrd, yPrd, zPrd, rPrd, dxdt, dydt, dzdt, drdt, dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
                   errors, aboveMinRadFlags.getRowPtr(0), env->getMinRad(), tfr, lbb, timeStep, numBubbles, useGasExchange);

        error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max, errors, numBubbles);

        if (error < env->getErrorTolerance() && timeStep < 0.1)
            timeStep *= 1.9;
        else if (error > env->getErrorTolerance())
            timeStep *= 0.5;

        ++numLoopsDone;

        if (numLoopsDone > 1000)
        {
            std::cout << "Done " << numLoopsDone << " loops, and error is " << error << std::endl;
            throw std::runtime_error("Error.");
        }
    } while (error > env->getErrorTolerance());

    updateData();

    ++integrationStep;
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    if (calculateEnergy)
        ElasticEnergy = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, energies, numBubbles);

    if (deleteSmallBubbles() || integrationStep % 50 == 0)
        updateCellsAndNeighbors();

    return numBubbles > env->getMinNumBubbles();
}

template <typename... Arguments>
void Simulator::resetValues(Arguments... args)
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    cudaLaunch(defaultPolicy, resetKernel, 0.0, numBubbles, args...);
}

template <typename... Arguments>
void Simulator::reorganizeValues(Arguments... args)
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    cudaLaunch(defaultPolicy, reorganizeKernel, numBubbles, args...);
}

void Simulator::generateBubbles()
{
    std::cout << "Starting to generate data for bubbles." << std::endl;

    const int rngSeed = env->getRngSeed();
    const double avgRad = env->getAvgRad();
    const double stdDevRad = env->getStdDevRad();
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *y = bubbleData.getRowPtr((size_t)BP::Y);
    double *z = bubbleData.getRowPtr((size_t)BP::Z);

    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);

    double *r = bubbleData.getRowPtr((size_t)BP::R);
    double *w = bubbleData.getRowPtr((size_t)BP::R_PRD);

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));

    CURAND_CALL(curandGenerateUniformDouble(generator, x, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, y, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, z, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, r, numBubbles, avgRad, stdDevRad));

    CURAND_CALL(curandDestroyGenerator(generator));

    ExecutionPolicy defaultPolicy(128, numBubbles);
    cudaLaunch(defaultPolicy, assignDataToBubbles,
               x, y, z, xPrd, yPrd, zPrd, r, w, aboveMinRadFlags.getRowPtr(0), givenNumBubblesPerDim, tfr, lbb, avgRad, env->getMinRad(), numBubbles);
}

void Simulator::updateCellsAndNeighbors()
{
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

    numPairs.setBytesToZero();

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *y = bubbleData.getRowPtr((size_t)BP::Y);
    double *z = bubbleData.getRowPtr((size_t)BP::Z);
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    ExecutionPolicy defaultPolicy(128, numBubbles);
    cudaLaunch(defaultPolicy, assignBubblesToCells,
               x, y, z, bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3), cellDim, numBubbles);

    int *cellIndices = bubbleCellIndices.getRowPtr(0);
    int *bubbleIndices = bubbleCellIndices.getRowPtr(1);

    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(2)),
                                    cellIndices,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(3)),
                                    bubbleIndices,
                                    numBubbles);

    cudaLaunch(defaultPolicy, findOffsets,
               cellIndices, offsets, numCells, numBubbles);

    cudaLaunch(defaultPolicy, findSizes,
               offsets, sizes, numCells, numBubbles);

    int sharedMemSizeInBytes = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Max, sizes, numCells);
    sharedMemSizeInBytes *= sharedMemSizeInBytes;
    sharedMemSizeInBytes *= 2;
    sharedMemSizeInBytes *= sizeof(int);
    const int maxNumSharedVals = sharedMemSizeInBytes / sizeof(int);
    assertMemBelowLimit(sharedMemSizeInBytes);
    assert(sharedMemSizeInBytes > 0 && "Zero bytes of shared memory reserved!");

    reorganizeValues(ReorganizeType::COPY_FROM_INDEX, bubbleIndices, bubbleIndices,
                     bubbleData.getRowPtr((size_t)BP::X), bubbleData.getRowPtr((size_t)BP::X_PRD),
                     bubbleData.getRowPtr((size_t)BP::Y), bubbleData.getRowPtr((size_t)BP::Y_PRD),
                     bubbleData.getRowPtr((size_t)BP::Z), bubbleData.getRowPtr((size_t)BP::Z_PRD),
                     bubbleData.getRowPtr((size_t)BP::R), bubbleData.getRowPtr((size_t)BP::R_PRD),
                     bubbleData.getRowPtr((size_t)BP::DXDT), bubbleData.getRowPtr((size_t)BP::DXDT_PRD),
                     bubbleData.getRowPtr((size_t)BP::DYDT), bubbleData.getRowPtr((size_t)BP::DYDT_PRD),
                     bubbleData.getRowPtr((size_t)BP::DZDT), bubbleData.getRowPtr((size_t)BP::DZDT_PRD),
                     bubbleData.getRowPtr((size_t)BP::DRDT), bubbleData.getRowPtr((size_t)BP::DRDT_PRD),
                     bubbleData.getRowPtr((size_t)BP::DXDT_OLD), bubbleData.getRowPtr((size_t)BP::ENERGY),
                     bubbleData.getRowPtr((size_t)BP::DYDT_OLD), bubbleData.getRowPtr((size_t)BP::FREE_AREA),
                     bubbleData.getRowPtr((size_t)BP::DZDT_OLD), bubbleData.getRowPtr((size_t)BP::ERROR),
                     bubbleData.getRowPtr((size_t)BP::DRDT_OLD), bubbleData.getRowPtr((size_t)BP::VOLUME));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(x),
                              static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)), sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(), cudaMemcpyDeviceToDevice));

    gridSize.z *= CUBBLE_NUM_NEIGHBORS + 1;
    assertGridSizeBelowLimit(gridSize);

    ExecutionPolicy findPolicy(256, numBubbles);
    findPolicy.gridSize = gridSize;
    findPolicy.sharedMemBytes = sharedMemSizeInBytes;
    cudaLaunch(findPolicy, findBubblePairs,
               x, y, z, r, offsets, sizes,
               neighborPairIndices.getRowPtr(2), neighborPairIndices.getRowPtr(3),
               numPairs.getDataPtr(), numCells, numBubbles, env->getTfr() - env->getLbb(),
               maxNumSharedVals, (int)neighborPairIndices.getWidth());

    CUDA_CALL(cudaMemcpy(&hostNumPairs, static_cast<void *>(numPairs.getDataPtr()), sizeof(int), cudaMemcpyDeviceToHost));

    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(neighborPairIndices.getRowPtr(2)),
                                    neighborPairIndices.getRowPtr(0),
                                    const_cast<const int *>(neighborPairIndices.getRowPtr(3)),
                                    neighborPairIndices.getRowPtr(1),
                                    hostNumPairs);
}

void Simulator::updateData()
{
    const size_t numBytesToCopy = 4 * sizeof(double) * bubbleData.getWidth();

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    double *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);

    CUDA_CALL(cudaMemcpyAsync(dxdtOld, dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpyAsync(x, xPrd, 2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
}

bool Simulator::deleteSmallBubbles()
{
    int *flag = aboveMinRadFlags.getRowPtr(0);
    const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Sum, flag, numBubbles);

    bool atLeastOneBubbleDeleted = numBubblesAboveMinRad < numBubbles;
    if (atLeastOneBubbleDeleted)
    {
        NVTX_RANGE_PUSH_A("BubbleRemoval");
        ExecutionPolicy defaultPolicy(128, numBubbles);

        double *r = bubbleData.getRowPtr((size_t)BP::R);
        double *volumes = bubbleData.getRowPtr((size_t)BP::VOLUME);

        // HACK: This is potentially very dangerous, if the used space is decreased in the future.
        double *volumeMultiplier = bubbleData.getRowPtr((size_t)BP::ERROR) + numBubblesAboveMinRad;
        cudaMemset(static_cast<void *>(volumeMultiplier), 0, sizeof(double));

        cudaLaunch(defaultPolicy, calculateRedistributedGasVolume,
                   volumes, r, flag, volumeMultiplier, env->getPi(), numBubbles);

        const double invTotalVolume = 1.0 / cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, volumes, numBubbles);

        int *newIdx = aboveMinRadFlags.getRowPtr(1);
        cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

        reorganizeValues(ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, flag,
                         bubbleData.getRowPtr((size_t)BP::X), bubbleData.getRowPtr((size_t)BP::X_PRD),
                         bubbleData.getRowPtr((size_t)BP::Y), bubbleData.getRowPtr((size_t)BP::Y_PRD),
                         bubbleData.getRowPtr((size_t)BP::Z), bubbleData.getRowPtr((size_t)BP::Z_PRD),
                         bubbleData.getRowPtr((size_t)BP::R), bubbleData.getRowPtr((size_t)BP::R_PRD),
                         bubbleData.getRowPtr((size_t)BP::DXDT), bubbleData.getRowPtr((size_t)BP::DXDT_PRD),
                         bubbleData.getRowPtr((size_t)BP::DYDT), bubbleData.getRowPtr((size_t)BP::DYDT_PRD),
                         bubbleData.getRowPtr((size_t)BP::DZDT), bubbleData.getRowPtr((size_t)BP::DZDT_PRD),
                         bubbleData.getRowPtr((size_t)BP::DRDT), bubbleData.getRowPtr((size_t)BP::DRDT_PRD),
                         bubbleData.getRowPtr((size_t)BP::DXDT_OLD), bubbleData.getRowPtr((size_t)BP::ENERGY),
                         bubbleData.getRowPtr((size_t)BP::DYDT_OLD), bubbleData.getRowPtr((size_t)BP::FREE_AREA),
                         bubbleData.getRowPtr((size_t)BP::DZDT_OLD), bubbleData.getRowPtr((size_t)BP::ERROR),
                         bubbleData.getRowPtr((size_t)BP::DRDT_OLD), bubbleData.getRowPtr((size_t)BP::VOLUME));
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X)),
                                  static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)), sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(), cudaMemcpyDeviceToDevice));

        numBubbles = numBubblesAboveMinRad;
        cudaLaunch(defaultPolicy, addVolume,
                   r, volumeMultiplier, numBubbles, invTotalVolume);

        NVTX_RANGE_POP();
    }

    return atLeastOneBubbleDeleted;
}

dim3 Simulator::getGridSize()
{
    int numBubblesPerCell = env->getNumBubblesPerCell();
#if (NUM_DIM == 3)
    int numCellsPerDim = std::ceil(std::cbrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, numCellsPerDim);
#else
    int numCellsPerDim = std::ceil(std::sqrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, 1);
#endif

    return gridSize;
}

double Simulator::getVolumeOfBubbles()
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    double *volPtr = bubbleData.getRowPtr((size_t)BP::VOLUME);
    cudaLaunch(defaultPolicy, calculateVolumes,
               r, volPtr, numBubbles, env->getPi());
    double volume = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, volPtr, numBubbles);

    return volume;
}

double Simulator::getAverageRadius()
{
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    double avgRad = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, r, numBubbles);
    avgRad /= numBubbles;

    return avgRad;
}

void Simulator::getBubbles(std::vector<Bubble> &bubbles) const
{
    bubbles.clear();
    bubbles.resize(numBubbles);

    size_t memoryStride = bubbleData.getWidth();
    double *devX = bubbleData.getRowPtr((size_t)BP::X);
    std::vector<double> xyzr;
    xyzr.resize(memoryStride * 4);

    CUDA_CALL(cudaMemcpy(xyzr.data(), devX, sizeof(double) * 4 * memoryStride, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numBubbles; ++i)
    {
        Bubble b;
        dvec pos(-1, -1, -1);
        pos.x = xyzr[i];
        pos.y = xyzr[i + memoryStride];
        pos.z = xyzr[i + 2 * memoryStride];
        b.setPos(pos);
        b.setRadius(xyzr[i + 3 * memoryStride]);
        bubbles[i] = b;
    }
}
} // namespace cubble