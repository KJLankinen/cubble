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
    bubbleData = DeviceArray<double>(numBubbles, (size_t)BP::NUM_VALUES);
    aboveMinRadFlags = DeviceArray<int>(numBubbles, 2);
    bubbleCellIndices = DeviceArray<int>(numBubbles, 4);
    neighbors = DeviceArray<int>(numBubbles, neighborStride + 1);

    const dim3 gridSize = getGridSize();
    size_t numCells = gridSize.x * gridSize.y * gridSize.z;
    cellData = DeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);

    hostData.resize(bubbleData.getSize(), 0);

    CUDA_CALL(cudaGetSymbolAddress((void **)&dtfa, dTotalFreeArea));
    assert(dtfa != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dtfapr, dTotalFreeAreaPerRadius));
    assert(dtfapr != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&mbpc, dMaxBubblesPerCell));
    assert(mbpc != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dvm, dVolumeMultiplier));
    assert(dvm != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dtv, dTotalVolume));
    assert(dtv != nullptr);

    CUDA_CALL(cudaStreamCreateWithFlags(&asyncCopyDDStream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&asyncCopyDHStream, cudaStreamNonBlocking));
    CUDA_CALL(cudaEventCreateWithFlags(&asyncCopyDDEvent, cudaEventBlockingSync));
    CUDA_CALL(cudaEventCreateWithFlags(&asyncCopyDHEvent, cudaEventBlockingSync));

    for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        neighborStreamVec.emplace_back();
        neighborEventVec.emplace_back();
        CUDA_CALL(cudaStreamCreateWithFlags(&neighborStreamVec[i], cudaStreamNonBlocking));
        CUDA_CALL(cudaEventCreate(&neighborEventVec[i]));
    }

    pinnedInt = PinnedHostArray<int>(1);
    pinnedDouble = PinnedHostArray<double>(1);

    printRelevantInfoOfCurrentDevice();
}

Simulator::~Simulator()
{
    CUDA_CALL(cudaStreamDestroy(asyncCopyDDStream));
    CUDA_CALL(cudaStreamDestroy(asyncCopyDHStream));
    CUDA_CALL(cudaEventDestroy(asyncCopyDDEvent));
    CUDA_CALL(cudaEventDestroy(asyncCopyDHEvent));

    for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        CUDA_CALL(cudaStreamDestroy(neighborStreamVec[i]));
        CUDA_CALL(cudaEventDestroy(neighborEventVec[i]));
    }
}

void Simulator::setupSimulation()
{
    generateBubbles();

    const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), numBubbles);
    if (numBubblesAboveMinRad < numBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

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

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const dvec interval = tfr - lbb;
    ExecutionPolicy defaultPolicy(128, numBubbles);

    double timeStep = env->getTimeStep();

    cudaLaunch(defaultPolicy, resetKernel,
               0.0, numBubbles,
               dxdtOld, dydtOld, dzdtOld, drdtOld);

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    cudaLaunch(defaultPolicy, velocityKernel,
               numBubbles, env->getFZeroPerMuZero(), neighbors.get(), r,
               interval.x, PBC_X == 1, x, dxdtOld,
               interval.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
               ,
               interval.z, PBC_Z == 1, z, dzdtOld
#endif
    );

    cudaLaunch(defaultPolicy, eulerKernel,
               numBubbles, timeStep,
               x, dxdtOld,
               y, dydtOld
#if (NUM_DIM == 3)
               ,
               z, dzdtOld
#endif
    );

    cudaLaunch(defaultPolicy, boundaryWrapKernel,
               numBubbles,
               x, lbb.x, tfr.x,
               y, lbb.y, tfr.y,
               z, lbb.z, tfr.z);

    cudaLaunch(defaultPolicy, resetKernel,
               0.0, numBubbles,
               dxdtOld, dydtOld, dzdtOld, drdtOld);

    cudaLaunch(defaultPolicy, velocityKernel,
               numBubbles, env->getFZeroPerMuZero(), neighbors.get(), r,
               interval.x, PBC_X == 1, x, dxdtOld,
               interval.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
               ,
               interval.z, PBC_Z == 1, z, dzdtOld
#endif
    );
}

bool Simulator::integrate(bool useGasExchange, bool calculateEnergy)
{
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy gasExchangePolicy(128, numBubbles);
    gasExchangePolicy.stream = asyncCopyDHStream;

    double timeStep = env->getTimeStep();

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

    int *flag = aboveMinRadFlags.getRowPtr(0);
    const dvec interval = env->getTfr() - env->getLbb();

    double error = 100000;
    size_t numLoopsDone = 0;
    do
    {
        NVTX_RANGE_PUSH_A("Integration step");

        cudaLaunch(defaultPolicy, resetKernel,
                   0.0, numBubbles,
                   dxdtPrd, dydtPrd, dzdtPrd, drdtPrd);

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

        cudaLaunch(defaultPolicy, velocityKernel,
                   numBubbles, env->getFZeroPerMuZero(), neighbors.get(), rPrd,
                   interval.x, PBC_X == 1, xPrd, dxdtPrd,
                   interval.y, PBC_Y == 1, yPrd, dydtPrd
#if (NUM_DIM == 3)
                   ,
                   interval.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
        );

        if (useGasExchange)
        {
            cudaLaunch(gasExchangePolicy, gasExchangeKernel,
                       numBubbles,
                       env->getPi(),
                       neighbors.get(),
                       rPrd,
                       drdtPrd,
                       freeArea,
                       errors,
                       interval.x, PBC_X == 1, xPrd, dxdtPrd,
                       interval.y, PBC_Y == 1, yPrd, dydtPrd
#if (NUM_DIM == 3)
                       ,
                       interval.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
            );

            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, errors, dtfapr, numBubbles, gasExchangePolicy.stream);
            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, freeArea, dtfa, numBubbles, gasExchangePolicy.stream);
            cudaLaunch(gasExchangePolicy, calculateFinalRadiusChangeRate,
                       drdtPrd, rPrd, freeArea, numBubbles, 1.0 / env->getPi(), env->getKappa(), env->getKParameter());
            
            CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, gasExchangePolicy.stream));
            CUDA_CALL(cudaStreamWaitEvent(0, asyncCopyDHEvent, 0));
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

        CUDA_CALL(cudaEventRecord(asyncCopyDHEvent));
        CUDA_CALL(cudaStreamWaitEvent(asyncCopyDHStream, asyncCopyDHEvent, 0));
        cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, errors, dtfa, numBubbles, asyncCopyDHStream);
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(double), cudaMemcpyDeviceToHost, asyncCopyDHStream));
        CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, asyncCopyDHStream));

        cudaLaunch(defaultPolicy, boundaryWrapKernel,
                   numBubbles,
                   xPrd, lbb.x, tfr.x,
                   yPrd, lbb.y, tfr.y,
                   zPrd, lbb.z, tfr.z);

        cudaLaunch(defaultPolicy, setFlagIfGreaterThanConstantKernel, numBubbles, flag, rPrd, env->getMinRad());

        CUDA_CALL(cudaEventRecord(asyncCopyDDEvent));
        CUDA_CALL(cudaStreamWaitEvent(asyncCopyDDStream, asyncCopyDDEvent, 0));
        cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, flag, static_cast<int *>(mbpc), numBubbles, asyncCopyDDStream);
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, asyncCopyDDStream));
        CUDA_CALL(cudaEventRecord(asyncCopyDDEvent, asyncCopyDDStream));

        CUDA_CALL(cudaEventSynchronize(asyncCopyDHEvent));
        error = pinnedDouble.get()[0];
        if (error < env->getErrorTolerance() && timeStep < 0.1)
            timeStep *= 1.9;
        else if (error > env->getErrorTolerance())
            timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > env->getErrorTolerance());

    updateData();

    ++integrationStep;
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    if (calculateEnergy)
    {
        cudaLaunch(defaultPolicy, potentialEnergyKernel,
                   numBubbles,
                   neighbors.get(),
                   r,
                   energies,
                   interval.x, PBC_X == 1, x, dxdt,
                   interval.y, PBC_Y == 1, y, dydt
#if (NUM_DIM == 3)
                   ,
                   interval.z, PBC_Z == 1, z, dzdt
#endif
        );
        ElasticEnergy = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, energies, numBubbles);
    }

    CUDA_CALL(cudaEventSynchronize(asyncCopyDDEvent));
    const int numBubblesAboveMinRad = pinnedInt.get()[0];
    const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;
    if (shouldDeleteBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    if (shouldDeleteBubbles || integrationStep % 50 == 0)
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

    neighbors.setBytesToZero();

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
    CUDA_CALL(cudaEventRecord(asyncCopyDHEvent));

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

    ExecutionPolicy findPolicy(128, numBubbles);
    findPolicy.gridSize = gridSize;

    dvec interval = env->getTfr() - env->getLbb();
    for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        findPolicy.stream = neighborStreamVec[i];
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], asyncCopyDDEvent, 0));
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], asyncCopyDHEvent, 0));
        cudaLaunch(findPolicy, neighborSearch,
                   i, neighborStride, numBubbles, numCells, neighbors.get(), offsets, sizes, r,
                   interval.x, PBC_X == 1, x,
                   interval.y, PBC_Y == 1, y
#if (NUM_DIM == 3)
                   ,
                   interval.z, PBC_Z == 1, z
#endif
        );

        CUDA_CALL(cudaEventRecord(neighborEventVec[i], neighborStreamVec[i]));
        CUDA_CALL(cudaStreamWaitEvent(0, neighborEventVec[i], 0));
    }
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

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
    NVTX_RANGE_PUSH_A("BubbleRemoval");
    ExecutionPolicy defaultPolicy(128, numBubbles);

    int *flag = aboveMinRadFlags.getRowPtr(0);
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    double *volumes = bubbleData.getRowPtr((size_t)BP::VOLUME);

    CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(double)));
    cudaLaunch(defaultPolicy, calculateRedistributedGasVolume,
               volumes, r, flag, env->getPi(), numBubbles);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, volumes, dtv, numBubbles);

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
                              static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)),
                              sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice));

    numBubbles = numBubblesAboveMinRad;
    cudaLaunch(defaultPolicy, addVolume, r, numBubbles);

    NVTX_RANGE_POP();
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