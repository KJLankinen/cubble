// -*- C++ -*-

#include "Simulator.h"
#include "Macros.h"
#include "Vec.h"
#include "Util.h"
#include "Kernels.h"

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

    bubbleData = FixedSizeDeviceArray<double>(numBubbles, (size_t)BubbleProperty::NUM_VALUES);
    aboveMinRadFlags = FixedSizeDeviceArray<int>(numBubbles, 2);
    indicesPerCell = FixedSizeDeviceArray<int>(numBubbles, 1);

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
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    generateBubbles();
    deleteSmallBubbles();
    updateCellsAndNeighbors();

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton perdictor-corrector method (ABMpc).

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);

    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT);

    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_OLD);

    double *energies = bubbleData.getRowPtr((size_t)BubbleProperty::ENERGY);
    double *freeArea = bubbleData.getRowPtr((size_t)BubbleProperty::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy accPolicy(128, hostNumPairs);

    double timeStep = env->getTimeStep();

    pointersToArrays.resize(4);
    pointersToArrays[0] = dxdtOld;
    pointersToArrays[1] = dydtOld;
    pointersToArrays[2] = dzdtOld;
    pointersToArrays[3] = drdtOld;
    resetValues();

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, energies, freeArea,
               firstIndices, secondIndices, numBubbles, hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), tfr - lbb, false, false);

    cudaLaunch(defaultPolicy, eulerIntegration,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, tfr, lbb, timeStep, numBubbles);

    if (deleteSmallBubbles())
        updateCellsAndNeighbors();

    resetValues();

    cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, energies, freeArea,
               firstIndices, secondIndices, numBubbles, hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), tfr - lbb, false, false);

    NVTX_RANGE_POP();
}

bool Simulator::integrate(bool useGasExchange, bool calculateEnergy)
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy accPolicy(128, hostNumPairs);

    double timeStep = env->getTimeStep();
    double error = 0;

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);

    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Z_PRD);
    double *rPrd = bubbleData.getRowPtr((size_t)BubbleProperty::R_PRD);

    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT);

    double *dxdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_PRD);
    double *dydtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_PRD);
    double *dzdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_PRD);
    double *drdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_PRD);

    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_OLD);

    double *energies = bubbleData.getRowPtr((size_t)BubbleProperty::ENERGY);
    double *errors = bubbleData.getRowPtr((size_t)BubbleProperty::ERROR);
    double *volumes = bubbleData.getRowPtr((size_t)BubbleProperty::VOLUME);
    double *freeArea = bubbleData.getRowPtr((size_t)BubbleProperty::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    size_t numLoopsDone = 0;
    do
    {
        pointersToArrays.resize(7);
        pointersToArrays[0] = dxdtPrd;
        pointersToArrays[1] = dydtPrd;
        pointersToArrays[2] = dzdtPrd;
        pointersToArrays[3] = drdtPrd;
        pointersToArrays[4] = freeArea;
        pointersToArrays[5] = energies;
        pointersToArrays[6] = errors;
        resetValues();

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

    NVTX_RANGE_POP();

    return numBubbles > env->getMinNumBubbles();
}

void Simulator::resetValues()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    // Using atomicAdd, so these need to be reset to 0 every time before use.
    // cudaMemset would be faster but it's not safe to assume that setting all bytes
    // of a double to zero means the double equals zero.
    // Some sort of preprocessing test might be possible to determine whether or not
    // cudaMemset(static_cast<void*>(doublePtr), 0, numBytesToReset) means double is actually zero.

    ExecutionPolicy defaultPolicy(128, numBubbles);

    for (size_t i = 0; i < pointersToArrays.size(); ++i)
    {
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        defaultPolicy.stream = stream;
        cudaLaunch(defaultPolicy, resetDoubleArrayToValue,
                   pointersToArrays[i], 0.0, numBubbles);
        CUDA_CALL(cudaStreamDestroy(stream));
    }

    NVTX_RANGE_POP();
}

void Simulator::generateBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    std::cout << "Starting to generate data for bubbles." << std::endl;

    const int rngSeed = env->getRngSeed();
    const double avgRad = env->getAvgRad();
    const double stdDevRad = env->getStdDevRad();
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);

    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Z_PRD);

    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double *w = bubbleData.getRowPtr((size_t)BubbleProperty::R_PRD);

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
    NVTX_RANGE_POP();
}

void Simulator::updateCellsAndNeighbors()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec domainDim(gridSize.x, gridSize.y, gridSize.z);
    ExecutionPolicy defaultPolicy(256, numBubbles);

    NVTX_RANGE_PUSH_A("Memsets");
    cellData.setBytesToZero();
    indicesPerCell.setBytesToZero();
    neighborPairIndices.setBytesToZero();
    numPairs.setBytesToZero();
    NVTX_RANGE_POP();

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    NVTX_RANGE_PUSH_A("Offsets");
    cudaLaunch(defaultPolicy, calculateOffsets,
               x, y, z, sizes, domainDim, numBubbles, numCells);
    NVTX_RANGE_POP();

    NVTX_RANGE_PUSH_A("Exclusive sum");
    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
    NVTX_RANGE_POP();

    NVTX_RANGE_PUSH_A("Memset sizes");
    CUDA_CALL(cudaMemset(static_cast<void *>(sizes), 0, sizeof(int) * numCells));
    NVTX_RANGE_POP();

    NVTX_RANGE_PUSH_A("Bubbles2Cells");
    cudaLaunch(defaultPolicy, bubblesToCells,
               x, y, z, indicesPerCell.getDataPtr(), offsets, sizes, domainDim, numBubbles);
    NVTX_RANGE_POP();

    gridSize.z *= CUBBLE_NUM_NEIGHBORS + 1;
    assertGridSizeBelowLimit(gridSize);

    NVTX_RANGE_PUSH_A("MaxNumCellRed");
    int sharedMemSizeInBytes = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Max, sizes, numCells);
    NVTX_RANGE_POP();

    sharedMemSizeInBytes *= sharedMemSizeInBytes;
    sharedMemSizeInBytes *= 2;
    sharedMemSizeInBytes *= sizeof(int);

    assertMemBelowLimit(sharedMemSizeInBytes);
    assert(sharedMemSizeInBytes > 0 && "Zero bytes of shared memory reserved!");

    defaultPolicy.gridSize = gridSize;
    defaultPolicy.sharedMemBytes = sharedMemSizeInBytes;
    NVTX_RANGE_PUSH_A("find");
    cudaLaunch(defaultPolicy, findBubblePairs,
               x, y, z, r, indicesPerCell.getDataPtr(), offsets, sizes,
               neighborPairIndices.getRowPtr(2), neighborPairIndices.getRowPtr(3),
               numPairs.getDataPtr(), numCells, numBubbles, env->getTfr() - env->getLbb(),
               sharedMemSizeInBytes / sizeof(int), neighborPairIndices.getWidth());
    NVTX_RANGE_POP();

    NVTX_RANGE_PUSH_A("mecpy numpairs");
    CUDA_CALL(cudaMemcpy(&hostNumPairs, static_cast<void *>(numPairs.getDataPtr()), sizeof(int), cudaMemcpyDeviceToHost));
    NVTX_RANGE_POP();

    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(neighborPairIndices.getRowPtr(2)),
                                    neighborPairIndices.getRowPtr(0),
                                    const_cast<const int *>(neighborPairIndices.getRowPtr(3)),
                                    neighborPairIndices.getRowPtr(1),
                                    hostNumPairs);

    NVTX_RANGE_POP();
}

void Simulator::updateData()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    // x, y, z, r are in memory continuously, so we can just make three copies with 4x the data of one component.
    size_t numBytesToCopy = 4 * sizeof(double) * bubbleData.getWidth();

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dxdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_PRD);
    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);

    cudaStream_t stream1, stream2;
    CUDA_CALL(cudaStreamCreate(&stream1));
    CUDA_CALL(cudaMemcpyAsync(x, xPrd, numBytesToCopy, cudaMemcpyDeviceToDevice, stream1));
    CUDA_CALL(cudaStreamDestroy(stream1));

    CUDA_CALL(cudaStreamCreate(&stream2));
    CUDA_CALL(cudaMemcpyAsync(dxdtOld, dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice, stream2));
    CUDA_CALL(cudaMemcpyAsync(dxdt, dxdtPrd, numBytesToCopy, cudaMemcpyDeviceToDevice, stream2));
    CUDA_CALL(cudaStreamDestroy(stream2));

    NVTX_RANGE_POP();
}

bool Simulator::deleteSmallBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    int *flag = aboveMinRadFlags.getRowPtr(0);
    const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Sum, flag, numBubbles);

    bool atLeastOneBubbleDeleted = numBubblesAboveMinRad < numBubbles;
    if (atLeastOneBubbleDeleted)
    {
        NVTX_RANGE_PUSH_A("BubbleRemoval");

        ExecutionPolicy defaultPolicy(128, numBubbles);

        double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
        double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
        double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
        double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);

        double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
        double *yPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Y_PRD);
        double *zPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Z_PRD);
        double *rPrd = bubbleData.getRowPtr((size_t)BubbleProperty::R_PRD);

        double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
        double *dydt = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT);
        double *dzdt = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT);
        double *drdt = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT);

        double *dxdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_PRD);
        double *dydtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_PRD);
        double *dzdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_PRD);
        double *drdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_PRD);

        double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
        double *dydtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_OLD);
        double *dzdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_OLD);
        double *drdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_OLD);

        double *energies = bubbleData.getRowPtr((size_t)BubbleProperty::ENERGY);
        double *errors = bubbleData.getRowPtr((size_t)BubbleProperty::ERROR);
        double *volumes = bubbleData.getRowPtr((size_t)BubbleProperty::VOLUME);
        double *freeArea = bubbleData.getRowPtr((size_t)BubbleProperty::FREE_AREA);

        // HACK: This is potentially very dangerous, if the used space is decreased in the future.
        double *volumeMultiplier = errors + numBubblesAboveMinRad;
        cudaMemset(static_cast<void *>(volumeMultiplier), 0, sizeof(double));

        cudaLaunch(defaultPolicy, calculateRedistributedGasVolume,
                   volumes, r, flag, volumeMultiplier, env->getPi(), numBubbles);

        int *newIdx = aboveMinRadFlags.getRowPtr(1);
        cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

        cudaLaunch(defaultPolicy, removeSmallBubbles,
                   xPrd, yPrd, zPrd, rPrd,
                   x, y, z, r,
                   dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
                   dxdt, dydt, dzdt, drdt,
                   energies, freeArea, errors, volumes,
                   dxdtOld, dydtOld, dzdtOld, drdtOld,
                   newIdx, flag, numBubbles);

        const size_t numBytesToCopy = 4 * sizeof(double) * bubbleData.getWidth();
        cudaStream_t stream1, stream2, stream3;
        CUDA_CALL(cudaStreamCreate(&stream1));
        CUDA_CALL(cudaStreamCreate(&stream2));
        CUDA_CALL(cudaStreamCreate(&stream3));
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpyAsync(x, xPrd, numBytesToCopy, cudaMemcpyDeviceToDevice, stream1));
        CUDA_CALL(cudaMemcpyAsync(dxdt, dxdtPrd, numBytesToCopy, cudaMemcpyDeviceToDevice, stream2));
        CUDA_CALL(cudaMemcpyAsync(dxdtOld, energies, numBytesToCopy, cudaMemcpyDeviceToDevice, stream3));

        CUDA_CALL(cudaStreamDestroy(stream1));
        CUDA_CALL(cudaStreamDestroy(stream2));
        CUDA_CALL(cudaStreamDestroy(stream3));

        numBubbles = numBubblesAboveMinRad;
        const double invTotalVolume = 1.0 / getVolumeOfBubbles();
        cudaLaunch(defaultPolicy, addVolume,
                   r, volumeMultiplier, numBubbles, invTotalVolume);

        NVTX_RANGE_POP();
    }

    NVTX_RANGE_POP();

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
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    ExecutionPolicy defaultPolicy(128, numBubbles);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double *volPtr = bubbleData.getRowPtr((size_t)BubbleProperty::VOLUME);
    cudaLaunch(defaultPolicy, calculateVolumes,
               r, volPtr, numBubbles, env->getPi());
    double volume = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, volPtr, numBubbles);

    NVTX_RANGE_POP();

    return volume;
}

double Simulator::getAverageRadius()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double avgRad = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, r, numBubbles);
    avgRad /= numBubbles;

    NVTX_RANGE_POP();

    return avgRad;
}

void Simulator::getBubbles(std::vector<Bubble> &bubbles) const
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    bubbles.clear();
    bubbles.resize(numBubbles);

    size_t memoryStride = bubbleData.getWidth();
    double *devX = bubbleData.getRowPtr((size_t)BubbleProperty::X);
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

    NVTX_RANGE_POP();
}
} // namespace cubble