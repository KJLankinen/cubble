// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
#include "Util.h"
#include "BubbleKernels.cuh"
#include "UtilityKernels.cuh"
#include "IntegrationKernels.cuh"

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
    dvec relDim = env->getBoxRelativeDimensions();
    relDim /= relDim.x;
    const float d = 2 * env->getAvgRad();
#if (NUM_DIM == 3)
    const float x = std::cbrt(env->getNumBubbles() * d * d * d / (relDim.y * relDim.z));
    dvec tfr = relDim * x;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), std::ceil(tfr.z / d));
    numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
#else
    const float x = std::sqrt(env->getNumBubbles() * d * d / relDim.y);
    dvec tfr = relDim * x;
    tfr.z = 0;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), 0);
    numBubbles = bubblesPerDim.x * bubblesPerDim.y;
#endif
    bubblesPerDimAtStart = bubblesPerDim;
    tfr = d * bubblesPerDim.asType<double>();
    env->setTfr(tfr + env->getLbb());

    cubWrapper = std::make_shared<CubWrapper>(env, numBubbles);
    bubbleData = FixedSizeDeviceArray<double>(numBubbles, (size_t)BP::NUM_VALUES);
    aboveMinRadFlags = FixedSizeDeviceArray<int>(numBubbles, 2);
    bubbleCellIndices = FixedSizeDeviceArray<int>(numBubbles, 4);

    // TODO: Figure out a more sensible value for this.
    const int maxNumPairs = (CUBBLE_NUM_NEIGHBORS + 1) * env->getNumBubblesPerCell() * numBubbles;
    neighborPairIndices = FixedSizeDeviceArray<int>(maxNumPairs, 2);
    numPairs = FixedSizeDeviceArray<int>(1, 1);

    const dim3 gridSize = getGridSize();
    size_t numCells = gridSize.x * gridSize.y * gridSize.z;
    cellData = FixedSizeDeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);

    hostData.resize(bubbleData.getSize(), 0);

    CUDA_CALL(cudaGetSymbolAddress((void **)&dtfa, dTotalFreeArea));
    assert(dtfa != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dtfapr, dTotalFreeAreaPerRadius));
    assert(dtfapr != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&mbpc, dMaxBubblesPerCell));
    assert(mbpc != nullptr);

    CUDA_CALL(cudaStreamCreateWithFlags(&asyncCopyDDStream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&asyncCopyDHStream, cudaStreamNonBlocking));
    CUDA_CALL(cudaEventCreateWithFlags(&asyncCopyDDEvent, cudaEventBlockingSync));
    CUDA_CALL(cudaEventCreateWithFlags(&asyncCopyDHEvent, cudaEventBlockingSync));

    pinnedInt = PinnedHostArray<int>(1);

    printRelevantInfoOfCurrentDevice();
}

Simulator::~Simulator()
{
    CUDA_CALL(cudaStreamDestroy(asyncCopyDDStream));
    CUDA_CALL(cudaStreamDestroy(asyncCopyDHStream));
    CUDA_CALL(cudaEventDestroy(asyncCopyDDEvent));
    CUDA_CALL(cudaEventDestroy(asyncCopyDHEvent));
}

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
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy accPolicy(128, hostNumPairs);

    double timeStep = env->getTimeStep();

    cudaLaunch(defaultPolicy, resetKernel,
               0.0, numBubbles,
               dxdtOld, dydtOld, dzdtOld, drdtOld);

    std::cout << "Calculating some initial values as a part of setup."
              << " Num bubbles: " << numBubbles
              << ", host num pairs: " << hostNumPairs
              << std::endl;

    cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
               x, y, z, r, dxdtOld, dydtOld, dzdtOld, drdtOld, energies, freeArea,
               firstIndices, secondIndices, numBubbles, hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), tfr - lbb, false, false);

    cudaLaunch(defaultPolicy, eulerKernel,
               numBubbles, timeStep,
               x, dxdtOld,
               y, dydtOld,
               z, dzdtOld);

    cudaLaunch(defaultPolicy, boundaryWrapKernel,
               numBubbles,
               x, lbb.x, tfr.x,
               y, lbb.y, tfr.y,
               z, lbb.z, tfr.z);

    cudaLaunch(defaultPolicy, resetKernel,
               0.0, numBubbles,
               dxdtOld, dydtOld, dzdtOld, drdtOld);

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
        NVTX_RANGE_PUSH_A("Integration step");

        cudaLaunch(defaultPolicy, resetKernel,
                   0.0, numBubbles,
                   dxdtPrd, dydtPrd, dzdtPrd, drdtPrd, freeArea, energies, errors);

        //HACK:  This is REALLY stupid, but doing it temporarily.
        if (useGasExchange)
            cudaLaunch(defaultPolicy, predictKernel,
                       numBubbles, timeStep,
                       xPrd, x, dxdt, dxdtOld,
                       yPrd, y, dydt, dydtOld,
                       zPrd, z, dzdt, dzdtOld,
                       rPrd, r, drdt, drdtOld);
        else
            cudaLaunch(defaultPolicy, predictKernel,
                       numBubbles, timeStep,
                       xPrd, x, dxdt, dxdtOld,
                       yPrd, y, dydt, dydtOld,
                       zPrd, z, dzdt, dzdtOld);

        cudaLaunch(accPolicy, calculateVelocityAndGasExchange,
                   xPrd, yPrd, zPrd, rPrd, dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
                   energies, freeArea, firstIndices, secondIndices, numBubbles,
                   hostNumPairs, env->getFZeroPerMuZero(), env->getPi(), env->getTfr() - env->getLbb(), calculateEnergy, useGasExchange);

        if (useGasExchange)
        {
            cudaLaunch(defaultPolicy, calculateFreeAreaPerRadius,
                       rPrd, freeArea, errors, env->getPi(), numBubbles);

            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, errors, dtfapr, numBubbles);
            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, freeArea, dtfa, numBubbles);
            cudaLaunch(defaultPolicy, calculateFinalRadiusChangeRate,
                       drdtPrd, rPrd, freeArea, numBubbles, 1.0 / env->getPi(), env->getKappa(), env->getKParameter());
        }

        //HACK:  This is REALLY stupid, but doing it temporarily.
        if (useGasExchange)
            cudaLaunch(defaultPolicy, correctKernel,
                       numBubbles, timeStep, errors,
                       xPrd, x, dxdt, dxdtPrd,
                       yPrd, y, dydt, dydtPrd,
                       zPrd, z, dzdt, dzdtPrd,
                       rPrd, r, drdt, drdtPrd);
        else
            cudaLaunch(defaultPolicy, correctKernel,
                       numBubbles, timeStep, errors,
                       xPrd, x, dxdt, dxdtPrd,
                       yPrd, y, dydt, dydtPrd,
                       zPrd, z, dzdt, dzdtPrd);

        error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max, errors, numBubbles);

        cudaLaunch(defaultPolicy, boundaryWrapKernel,
                   numBubbles,
                   xPrd, lbb.x, tfr.x,
                   yPrd, lbb.y, tfr.y,
                   zPrd, lbb.z, tfr.z);

        cudaLaunch(defaultPolicy, setFlagIfGreaterThanConstantKernel,
                   numBubbles, aboveMinRadFlags.getRowPtr(0), rPrd, env->getMinRad());

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
        NVTX_RANGE_POP();
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
    assert(bubblesPerDimAtStart.x > 0);
    assert(bubblesPerDimAtStart.y > 0);
#if (NUM_DIM == 3)
    assert(bubblesPerDimAtStart.z > 0);
#endif
    cudaLaunch(defaultPolicy, assignDataToBubbles,
               x, y, z, xPrd, yPrd, zPrd, r, w, aboveMinRadFlags.getRowPtr(0), bubblesPerDimAtStart, tfr, lbb, avgRad, env->getMinRad(), numBubbles);
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
    ExecutionPolicy asyncCopyDDPolicy(128, numBubbles, 0, asyncCopyDDStream);
    cudaLaunch(defaultPolicy, assignBubblesToCells,
               x, y, z, bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3), env->getLbb(), env->getTfr(), cellDim, numBubbles);

    int *cellIndices = bubbleCellIndices.getRowPtr(0);
    int *bubbleIndices = bubbleCellIndices.getRowPtr(1);

    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(2)),
                                    cellIndices,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(3)),
                                    bubbleIndices,
                                    numBubbles);

    CUDA_CALL(cudaEventRecord(asyncCopyDDEvent));
    CUDA_CALL(cudaStreamWaitEvent(asyncCopyDDStream, asyncCopyDDEvent, 0));

    cudaLaunch(defaultPolicy, findOffsets,
               cellIndices, offsets, numCells, numBubbles);

    cudaLaunch(defaultPolicy, findSizes,
               offsets, sizes, numCells, numBubbles);

    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Max, sizes, mbpc, numCells);
    CUDA_CALL(cudaEventRecord(asyncCopyDHEvent));
    CUDA_CALL(cudaStreamWaitEvent(asyncCopyDHStream, asyncCopyDHEvent, 0));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, asyncCopyDHStream));
    CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, asyncCopyDHStream));

    cudaLaunch(asyncCopyDDPolicy, reorganizeKernel,
               numBubbles, ReorganizeType::COPY_FROM_INDEX, bubbleIndices, bubbleIndices,
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
                              static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)),
                              sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice,
                              asyncCopyDDStream));

    CUDA_CALL(cudaEventRecord(asyncCopyDDEvent, asyncCopyDDStream));

    CUDA_CALL(cudaEventSynchronize(asyncCopyDHEvent));
    int sharedMemSizeInBytes = pinnedInt.get()[0];
    sharedMemSizeInBytes *= sharedMemSizeInBytes;
    sharedMemSizeInBytes *= 2;
    const int maxNumSharedVals = sharedMemSizeInBytes;
    sharedMemSizeInBytes *= sizeof(int);
    assertMemBelowLimit(sharedMemSizeInBytes);
    assert(sharedMemSizeInBytes > 0 && "Zero bytes of shared memory reserved!");

    gridSize.z *= CUBBLE_NUM_NEIGHBORS + 1;
    assertGridSizeBelowLimit(gridSize);

    ExecutionPolicy findPolicy(256, numBubbles);
    findPolicy.gridSize = gridSize;
    findPolicy.sharedMemBytes = sharedMemSizeInBytes;

    CUDA_CALL(cudaStreamWaitEvent(0, asyncCopyDDEvent, 0));
    CUDA_CALL(cudaStreamWaitEvent(0, asyncCopyDHEvent, 0));
    cudaLaunch(findPolicy, findBubblePairs,
               x, y, z, r, offsets, sizes,
               neighborPairIndices.getRowPtr(0), neighborPairIndices.getRowPtr(1),
               numPairs.get(), numCells, numBubbles, env->getTfr() - env->getLbb(),
               maxNumSharedVals, (int)neighborPairIndices.getWidth());

    CUDA_CALL(cudaMemcpy(&hostNumPairs, static_cast<void *>(numPairs.get()), sizeof(int), cudaMemcpyDeviceToHost));
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
    // This one copy is what keep this from going...
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

        cudaLaunch(defaultPolicy, reorganizeKernel,
                   numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, flag,
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