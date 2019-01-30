// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
#include "BubbleKernels.cuh"
#include "UtilityKernels.cuh"
#include "IntegrationKernels.cuh"

#include "cub.cuh"
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
    fpvec relDim = env->getBoxRelativeDimensions();
    relDim /= relDim.x;
    const CubbleFloatType d = 2 * env->getAvgRad();
#if (NUM_DIM == 3)
    const CubbleFloatType x = std::cbrt(env->getNumBubbles() * d * d * d / (relDim.y * relDim.z));
    fpvec tfr = relDim * x;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), std::ceil(tfr.z / d));
    numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
#else
    const CubbleFloatType x = std::sqrt(env->getNumBubbles() * d * d / relDim.y);
    fpvec tfr = relDim * x;
    tfr.z = 0;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), 0);
    numBubbles = bubblesPerDim.x * bubblesPerDim.y;
#endif
    bubblesPerDimAtStart = bubblesPerDim;
    tfr = d * bubblesPerDim.asType<CubbleFloatType>();
    env->setTfr(tfr + env->getLbb());

    cubWrapper = std::make_shared<CubWrapper>(env, numBubbles);
    bubbleData = DeviceArray<CubbleFloatType>(numBubbles, (size_t)BP::NUM_VALUES);
    aboveMinRadFlags = DeviceArray<int>(numBubbles, 2);
    bubbleCellIndices = DeviceArray<int>(numBubbles, 4);
    pairs = DeviceArray<int>(8 * numBubbles, 4);

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
    CUDA_CALL(cudaGetSymbolAddress((void **)&np, dNumPairs));
    assert(np != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dir, dInvRho));
    assert(dir != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dta, dTotalArea));
    assert(dta != nullptr);
    CUDA_CALL(cudaGetSymbolAddress((void **)&dasai, dAverageSurfaceAreaIn));
    assert(dasai != nullptr);

    CUDA_CALL(cudaStreamCreateWithFlags(&nonBlockingStream1, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&nonBlockingStream2, cudaStreamNonBlocking));
    CUDA_CALL(cudaEventCreateWithFlags(&blockingEvent1, cudaEventBlockingSync));
    CUDA_CALL(cudaEventCreateWithFlags(&blockingEvent2, cudaEventBlockingSync));

    for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        neighborStreamVec.emplace_back();
        neighborEventVec.emplace_back();
        CUDA_CALL(cudaStreamCreateWithFlags(&neighborStreamVec[i], cudaStreamNonBlocking));
        CUDA_CALL(cudaEventCreate(&neighborEventVec[i]));
    }

    pinnedInt = PinnedHostArray<int>(1);
    pinnedDouble = PinnedHostArray<CubbleFloatType>(1);

    printRelevantInfoOfCurrentDevice();
}

Simulator::~Simulator()
{
    CUDA_CALL(cudaStreamDestroy(nonBlockingStream1));
    CUDA_CALL(cudaStreamDestroy(nonBlockingStream2));
    CUDA_CALL(cudaEventDestroy(blockingEvent1));
    CUDA_CALL(cudaEventDestroy(blockingEvent2));

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

    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *y = bubbleData.getRowPtr((size_t)BP::Y);
    CubbleFloatType *z = bubbleData.getRowPtr((size_t)BP::Z);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);

    CubbleFloatType *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    CubbleFloatType *dydt = bubbleData.getRowPtr((size_t)BP::DYDT);
    CubbleFloatType *dzdt = bubbleData.getRowPtr((size_t)BP::DZDT);
    CubbleFloatType *drdt = bubbleData.getRowPtr((size_t)BP::DRDT);

    CubbleFloatType *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    CubbleFloatType *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    CubbleFloatType *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    CubbleFloatType *drdtOld = bubbleData.getRowPtr((size_t)BP::DRDT_OLD);

    CubbleFloatType *energies = bubbleData.getRowPtr((size_t)BP::ENERGY);
    CubbleFloatType *freeArea = bubbleData.getRowPtr((size_t)BP::FREE_AREA);

    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();
    const fpvec interval = tfr - lbb;
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    CubbleFloatType timeStep = env->getTimeStep();

    CUDA_LAUNCH(resetKernel, defaultPolicy,
                (CubbleFloatType)0.0, numBubbles,
                dxdtOld, dydtOld, dzdtOld, drdtOld);

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    CUDA_LAUNCH(velocityPairKernel, pairPolicy,
                env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
                interval.x, lbb.x, PBC_X == 1, x, dxdtOld,
                interval.y, lbb.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
                ,
                interval.z, lbb.z, PBC_Z == 1, z, dzdtOld
#endif
    );

    CUDA_LAUNCH(eulerKernel, defaultPolicy,
                numBubbles, timeStep,
                x, dxdtOld,
                y, dydtOld
#if (NUM_DIM == 3)
                ,
                z, dzdtOld
#endif
    );

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    CUDA_LAUNCH(boundaryWrapKernel, defaultPolicy,
                numBubbles
#if (PBC_X == 1)
                ,
                x, lbb.x, tfr.x
#endif
#if (PBC_Y == 1)
                ,
                y, lbb.y, tfr.y
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                ,
                z, lbb.z, tfr.z
#endif
    );
#endif

    CUDA_LAUNCH(resetKernel, defaultPolicy,
                (CubbleFloatType)0.0, numBubbles,
                dxdtOld, dydtOld, dzdtOld, drdtOld);

    CUDA_LAUNCH(velocityPairKernel, pairPolicy,
                env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
                interval.x, lbb.x, PBC_X == 1, x, dxdtOld,
                interval.y, lbb.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
                ,
                interval.z, lbb.z, PBC_Z == 1, z, dzdtOld
#endif
    );
}

bool Simulator::integrate(bool useGasExchange)
{
    ExecutionPolicy defaultPolicy(128, numBubbles);

    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    CubbleFloatType timeStep = env->getTimeStep();
    CubbleFloatType error = 100000;
    size_t numLoopsDone = 0;

    do
    {
        NVTX_RANGE_PUSH_A("Integration step");

        doReset(defaultPolicy);
        doPrediction(defaultPolicy, timeStep, useGasExchange, blockingEvent2);
        doVelocity(pairPolicy);
        if (useGasExchange)
            doGasExchange(pairPolicy, blockingEvent2, pairPolicy.stream);
        doCorrection(defaultPolicy, timeStep, useGasExchange, nonBlockingStream2);

        error = doError();
        if (error < env->getErrorTolerance() && timeStep < 0.1)
            timeStep *= 1.9;
        else if (error > env->getErrorTolerance())
            timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > env->getErrorTolerance());

    doBoundaryWrap(defaultPolicy);
    doBubbleSizeChecks(defaultPolicy, nonBlockingStream1, blockingEvent1);
    updateData();

    ++integrationStep;
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    CUDA_CALL(cudaEventSynchronize(blockingEvent1));

    const int numBubblesAboveMinRad = pinnedInt.get()[0];
    const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;

    if (shouldDeleteBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    if (shouldDeleteBubbles || integrationStep % 50 == 0)
        updateCellsAndNeighbors();

    bool continueSimulation = numBubbles > env->getMinNumBubbles();

    maxBubbleRadius = pinnedDouble.get()[0];
#if (NUM_DIM == 3)
    continueSimulation &= maxBubbleRadius < 0.5 * (env->getTfr() - env->getLbb()).getMinComponent();
#endif

    return continueSimulation;
}

void Simulator::doPrediction(const ExecutionPolicy &policy, CubbleFloatType timeStep, bool useGasExchange, cudaEvent_t &eventToMark)
{
    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *y = bubbleData.getRowPtr((size_t)BP::Y);
    CubbleFloatType *z = bubbleData.getRowPtr((size_t)BP::Z);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    CubbleFloatType *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);

    CubbleFloatType *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    CubbleFloatType *dydt = bubbleData.getRowPtr((size_t)BP::DYDT);
    CubbleFloatType *dzdt = bubbleData.getRowPtr((size_t)BP::DZDT);
    CubbleFloatType *drdt = bubbleData.getRowPtr((size_t)BP::DRDT);

    CubbleFloatType *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    CubbleFloatType *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    CubbleFloatType *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    CubbleFloatType *drdtOld = bubbleData.getRowPtr((size_t)BP::DRDT_OLD);

    if (useGasExchange)
    {
        CUDA_LAUNCH(predictKernel, policy,
                    numBubbles, timeStep,
                    xPrd, x, dxdt, dxdtOld,
                    yPrd, y, dydt, dydtOld,
#if (NUM_DIM == 3)
                    zPrd, z, dzdt, dzdtOld,
#endif
                    rPrd, r, drdt, drdtOld);
    }
    else
    {
        CUDA_LAUNCH(predictKernel, policy,
                    numBubbles, timeStep,
                    xPrd, x, dxdt, dxdtOld,
                    yPrd, y, dydt, dydtOld
#if (NUM_DIM == 3)
                    ,
                    zPrd, z, dzdt, dzdtOld
#endif
        );
    }

    CUDA_CALL(cudaEventRecord(eventToMark, policy.stream));
}

void Simulator::doCorrection(const ExecutionPolicy &policy, CubbleFloatType timeStep, bool useGasExchange, cudaStream_t &streamThatShouldWait)
{
    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *y = bubbleData.getRowPtr((size_t)BP::Y);
    CubbleFloatType *z = bubbleData.getRowPtr((size_t)BP::Z);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    CubbleFloatType *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);

    CubbleFloatType *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    CubbleFloatType *dydt = bubbleData.getRowPtr((size_t)BP::DYDT);
    CubbleFloatType *dzdt = bubbleData.getRowPtr((size_t)BP::DZDT);
    CubbleFloatType *drdt = bubbleData.getRowPtr((size_t)BP::DRDT);

    CubbleFloatType *dxdtPrd = bubbleData.getRowPtr((size_t)BP::DXDT_PRD);
    CubbleFloatType *dydtPrd = bubbleData.getRowPtr((size_t)BP::DYDT_PRD);
    CubbleFloatType *dzdtPrd = bubbleData.getRowPtr((size_t)BP::DZDT_PRD);
    CubbleFloatType *drdtPrd = bubbleData.getRowPtr((size_t)BP::DRDT_PRD);

    CubbleFloatType *errors = bubbleData.getRowPtr((size_t)BP::ERROR);

    if (useGasExchange)
    {
        CUDA_LAUNCH(correctKernel, policy,
                    numBubbles, timeStep, errors,
                    xPrd, x, dxdt, dxdtPrd,
                    yPrd, y, dydt, dydtPrd,
#if (NUM_DIM == 3)
                    zPrd, z, dzdt, dzdtPrd,
#endif
                    rPrd, r, drdt, drdtPrd);
    }
    else
    {
        CUDA_LAUNCH(correctKernel, policy,
                    numBubbles, timeStep, errors,
                    xPrd, x, dxdt, dxdtPrd,
                    yPrd, y, dydt, dydtPrd
#if (NUM_DIM == 3)
                    ,
                    zPrd, z, dzdt, dzdtPrd
#endif
        );
    }

    CUDA_CALL(cudaEventRecord(blockingEvent2, policy.stream));
    CUDA_CALL(cudaStreamWaitEvent(streamThatShouldWait, blockingEvent2, 0));
}

void Simulator::doGasExchange(ExecutionPolicy policy, const cudaEvent_t &eventToWaitOn, cudaStream_t &streamThatShouldWait)
{
    ExecutionPolicy gasExchangePolicy(128, numBubbles);
    gasExchangePolicy.stream = nonBlockingStream2;
    policy.stream = gasExchangePolicy.stream;

    CUDA_CALL(cudaStreamWaitEvent(gasExchangePolicy.stream, eventToWaitOn, 0));

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    CubbleFloatType *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    CubbleFloatType *drdtPrd = bubbleData.getRowPtr((size_t)BP::DRDT_PRD);
    CubbleFloatType *errors = bubbleData.getRowPtr((size_t)BP::ERROR);
    CubbleFloatType *freeArea = bubbleData.getRowPtr((size_t)BP::FREE_AREA);
    CubbleFloatType *volume = bubbleData.getRowPtr((size_t)BP::VOLUME);

    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();
    const fpvec interval = tfr - lbb;

    CUDA_LAUNCH(gasExchangeKernel, policy,
                numBubbles,
                env->getPi(),
                pairs.getRowPtr(0),
                pairs.getRowPtr(1),
                rPrd,
                drdtPrd,
                freeArea,
                interval.x, PBC_X == 1, xPrd,
                interval.y, PBC_Y == 1, yPrd
#if (NUM_DIM == 3)
                ,
                interval.z, PBC_Z == 1, zPrd
#endif
    );

    CUDA_LAUNCH(freeAreaKernel, gasExchangePolicy,
                numBubbles, env->getPi(), rPrd, freeArea, errors, volume);

    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, errors, dtfapr, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, freeArea, dtfa, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, volume, dta, numBubbles, gasExchangePolicy.stream);

    CUDA_LAUNCH(finalRadiusChangeRateKernel, gasExchangePolicy,
                drdtPrd, rPrd, freeArea, numBubbles, (CubbleFloatType)1.0 / env->getPi(), env->getKappa(), env->getKParameter());

    CUDA_CALL(cudaEventRecord(blockingEvent2, gasExchangePolicy.stream));
    CUDA_CALL(cudaStreamWaitEvent(streamThatShouldWait, blockingEvent2, 0));
}

void Simulator::doVelocity(const ExecutionPolicy &policy)
{
    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();
    const fpvec interval = tfr - lbb;

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    CubbleFloatType *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    CubbleFloatType *dxdtPrd = bubbleData.getRowPtr((size_t)BP::DXDT_PRD);
    CubbleFloatType *dydtPrd = bubbleData.getRowPtr((size_t)BP::DYDT_PRD);
    CubbleFloatType *dzdtPrd = bubbleData.getRowPtr((size_t)BP::DZDT_PRD);

    CUDA_LAUNCH(velocityPairKernel, policy,
                env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), rPrd,
                interval.x, lbb.x, PBC_X == 1, xPrd, dxdtPrd,
                interval.y, lbb.y, PBC_Y == 1, yPrd, dydtPrd
#if (NUM_DIM == 3)
                ,
                interval.z, lbb.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
    );

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    CUDA_LAUNCH(velocityWallKernel, policy,
                numBubbles, env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), rPrd
#if (PBC_X == 0)
                ,
                interval.x, lbb.x, PBC_X == 1, xPrd, dxdtPrd
#endif
#if (PBC_Y == 0)
                ,
                interval.y, lbb.y, PBC_Y == 1, yPrd, dydtPrd
#endif
#if (NUM_DIM == 3 && PBC_Z == 0)
                ,
                interval.z, lbb.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
    );
#endif
}

void Simulator::doReset(const ExecutionPolicy &policy)
{
    CUDA_LAUNCH(resetKernel, policy,
                (CubbleFloatType)0.0, numBubbles,
                bubbleData.getRowPtr((size_t)BP::DXDT_PRD),
                bubbleData.getRowPtr((size_t)BP::DYDT_PRD),
                bubbleData.getRowPtr((size_t)BP::DZDT_PRD),
                bubbleData.getRowPtr((size_t)BP::DRDT_PRD),
                bubbleData.getRowPtr((size_t)BP::FREE_AREA),
                bubbleData.getRowPtr((size_t)BP::ENERGY));
}

CubbleFloatType Simulator::doError()
{
    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Max, bubbleData.getRowPtr((size_t)BP::ERROR), dtfa, numBubbles, nonBlockingStream2);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(CubbleFloatType), cudaMemcpyDeviceToHost, nonBlockingStream2));
    CUDA_CALL(cudaEventRecord(blockingEvent2, nonBlockingStream2));
    CUDA_CALL(cudaEventSynchronize(blockingEvent2));

    return pinnedDouble.get()[0];
}

void Simulator::doBoundaryWrap(const ExecutionPolicy &policy)
{
    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    CUDA_LAUNCH(boundaryWrapKernel, policy,
                numBubbles
#if (PBC_X == 1)
                ,
                xPrd, lbb.x, tfr.x
#endif
#if (PBC_Y == 1)
                ,
                yPrd, lbb.y, tfr.y
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                ,
                zPrd, lbb.z, tfr.z
#endif
    );
#endif
}

void Simulator::doBubbleSizeChecks(ExecutionPolicy policy, cudaStream_t &streamToUse, cudaEvent_t &eventToMark)
{
    CubbleFloatType *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    policy.stream = streamToUse;

    CUDA_LAUNCH(setFlagIfGreaterThanConstantKernel, policy,
                numBubbles,
                aboveMinRadFlags.getRowPtr(0),
                rPrd,
                env->getMinRad());

    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, streamToUse));

    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Max, rPrd, static_cast<CubbleFloatType *>(dtfa), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(CubbleFloatType), cudaMemcpyDeviceToHost, streamToUse));

    CUDA_CALL(cudaEventRecord(eventToMark, streamToUse));
}

void Simulator::generateBubbles()
{
    std::cout << "Starting to generate data for bubbles." << std::endl;

    const int rngSeed = env->getRngSeed();
    const CubbleFloatType avgRad = env->getAvgRad();
    const CubbleFloatType stdDevRad = env->getStdDevRad();
    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();

    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *y = bubbleData.getRowPtr((size_t)BP::Y);
    CubbleFloatType *z = bubbleData.getRowPtr((size_t)BP::Z);

    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    CubbleFloatType *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);

    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);
    CubbleFloatType *w = bubbleData.getRowPtr((size_t)BP::R_PRD);

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));

#if (CUBBLE_FLOAT_TYPE == CUBBLE_FLOAT)
    CURAND_CALL(curandGenerateUniform(generator, x, numBubbles));
    CURAND_CALL(curandGenerateUniform(generator, y, numBubbles));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniform(generator, z, numBubbles));
#endif
    CURAND_CALL(curandGenerateUniform(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormal(generator, r, numBubbles, avgRad, stdDevRad));
#else
    CURAND_CALL(curandGenerateUniformDouble(generator, x, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, y, numBubbles));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, z, numBubbles));
#endif
    CURAND_CALL(curandGenerateUniformDouble(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, r, numBubbles, avgRad, stdDevRad));
#endif

    CURAND_CALL(curandDestroyGenerator(generator));

    ExecutionPolicy defaultPolicy(128, numBubbles);
    assert(bubblesPerDimAtStart.x > 0);
    assert(bubblesPerDimAtStart.y > 0);
#if (NUM_DIM == 3)
    assert(bubblesPerDimAtStart.z > 0);
#endif
    CUDA_LAUNCH(assignDataToBubbles, defaultPolicy,
                x, y, z, xPrd, yPrd, zPrd, r, w,
                aboveMinRadFlags.getRowPtr(0), bubblesPerDimAtStart,
                tfr, lbb, avgRad, env->getMinRad(), env->getPi(), numBubbles);

    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, w, dasai, numBubbles, defaultPolicy.stream);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(w), static_cast<void *>(r),
                              sizeof(CubbleFloatType) * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice,
                              defaultPolicy.stream));
}

void Simulator::updateCellsAndNeighbors()
{
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *y = bubbleData.getRowPtr((size_t)BP::Y);
    CubbleFloatType *z = bubbleData.getRowPtr((size_t)BP::Z);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);
    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    cellData.setBytesToZero();
    bubbleCellIndices.setBytesToZero();

    ExecutionPolicy defaultPolicy = {};
    defaultPolicy.blockSize = dim3(128, 1, 1);
    defaultPolicy.gridSize = dim3(256, 1, 1);
    ExecutionPolicy asyncCopyDDPolicy(128, numBubbles, 0, nonBlockingStream1);
    CUDA_LAUNCH(assignBubblesToCells, defaultPolicy,
                x, y, z, bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3), env->getLbb(), env->getTfr(), cellDim, numBubbles);

    int *cellIndices = bubbleCellIndices.getRowPtr(0);
    int *bubbleIndices = bubbleCellIndices.getRowPtr(1);

    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(2)),
                                    cellIndices,
                                    const_cast<const int *>(bubbleCellIndices.getRowPtr(3)),
                                    bubbleIndices,
                                    numBubbles);

    CUDA_CALL(cudaEventRecord(blockingEvent1));
    CUDA_CALL(cudaStreamWaitEvent(nonBlockingStream1, blockingEvent1, 0));

    cubWrapper->histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven,
                                                bubbleCellIndices.getRowPtr(2),
                                                sizes,
                                                numCells + 1,
                                                0,
                                                numCells,
                                                numBubbles);

    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
    CUDA_CALL(cudaEventRecord(blockingEvent2));

    CUDA_LAUNCH(reorganizeKernel, asyncCopyDDPolicy,
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
                              sizeof(CubbleFloatType) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice,
                              nonBlockingStream1));

    CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream1));

    fpvec interval = env->getTfr() - env->getLbb();

    ExecutionPolicy findPolicy;
    findPolicy.blockSize = dim3(128, 1, 1);
    findPolicy.gridSize = gridSize;
    findPolicy.sharedMemBytes = 0;

    CUDA_CALL(cudaMemset(np, 0, sizeof(int)));

    for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        findPolicy.stream = neighborStreamVec[i];
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], blockingEvent1, 0));
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], blockingEvent2, 0));
        CUDA_LAUNCH(neighborSearch, findPolicy,
                    i, numBubbles, numCells, static_cast<int>(pairs.getWidth()),
                    offsets, sizes, pairs.getRowPtr(2), pairs.getRowPtr(3), r,
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

    CUDA_CALL(cudaMemcpy(static_cast<void *>(pinnedInt.get()), np, sizeof(int), cudaMemcpyDeviceToHost));
    int numPairs = pinnedInt.get()[0];
    cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
                                    const_cast<const int *>(pairs.getRowPtr(2)),
                                    pairs.getRowPtr(0),
                                    const_cast<const int *>(pairs.getRowPtr(3)),
                                    pairs.getRowPtr(1),
                                    numPairs);
}

void Simulator::updateData()
{
    CUDA_CALL(cudaStreamWaitEvent(0, blockingEvent1, 0));
    const size_t numBytesToCopy = 4 * sizeof(CubbleFloatType) * bubbleData.getWidth();

    CubbleFloatType *x = bubbleData.getRowPtr((size_t)BP::X);
    CubbleFloatType *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    CubbleFloatType *dxdt = bubbleData.getRowPtr((size_t)BP::DXDT);
    CubbleFloatType *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);

    CUDA_CALL(cudaMemcpyAsync(dxdtOld, dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpyAsync(x, xPrd, 2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
}

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
    NVTX_RANGE_PUSH_A("BubbleRemoval");
    ExecutionPolicy defaultPolicy(128, numBubbles);

    int *flag = aboveMinRadFlags.getRowPtr(0);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);
    CubbleFloatType *volumes = bubbleData.getRowPtr((size_t)BP::VOLUME);

    CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(CubbleFloatType)));
    CUDA_LAUNCH(calculateRedistributedGasVolume, defaultPolicy,
                volumes, r, flag, env->getPi(), numBubbles);

    cubWrapper->reduceNoCopy<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, volumes, dtv, numBubbles);

    int *newIdx = aboveMinRadFlags.getRowPtr(1);
    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

    CUDA_LAUNCH(reorganizeKernel, defaultPolicy,
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
                              sizeof(CubbleFloatType) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice));

    numBubbles = numBubblesAboveMinRad;
    CUDA_LAUNCH(addVolume, defaultPolicy, r, numBubbles);

    NVTX_RANGE_POP();
}

dim3 Simulator::getGridSize()
{
    const int totalNumCells = std::ceil((CubbleFloatType)numBubbles / env->getNumBubblesPerCell());
    fpvec interval = env->getTfr() - env->getLbb();
    interval /= interval.x;
#if (NUM_DIM == 3)
    CubbleFloatType nx = std::cbrt((CubbleFloatType)totalNumCells / (interval.y * interval.z));
#else
    CubbleFloatType nx = std::sqrt((CubbleFloatType)totalNumCells / interval.y);
    interval.z = 0;
#endif
    ivec grid = (nx * interval).floor() + 1;
    assert(grid.x > 0);
    assert(grid.y > 0);
    assert(grid.z > 0);

    return dim3(grid.x, grid.y, grid.z);
}

void Simulator::calculateEnergy()
{
    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    const fpvec tfr = env->getTfr();
    const fpvec lbb = env->getLbb();
    const fpvec interval = tfr - lbb;

    CUDA_LAUNCH(potentialEnergyKernel, pairPolicy,
                numBubbles,
                pairs.getRowPtr(0),
                pairs.getRowPtr(1),
                bubbleData.getRowPtr((size_t)BP::R),
                bubbleData.getRowPtr((size_t)BP::ENERGY),
                interval.x, PBC_X == 1, bubbleData.getRowPtr((size_t)BP::X),
                interval.y, PBC_Y == 1, bubbleData.getRowPtr((size_t)BP::Y)
#if (NUM_DIM == 3)
                                            ,
                interval.z, PBC_Z == 1, bubbleData.getRowPtr((size_t)BP::Z)
#endif
    );

    ElasticEnergy = cubWrapper->reduce<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum,
                                                                                              bubbleData.getRowPtr((size_t)BP::ENERGY),
                                                                                              numBubbles);
}

CubbleFloatType Simulator::getVolumeOfBubbles()
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);
    CubbleFloatType *volPtr = bubbleData.getRowPtr((size_t)BP::VOLUME);
    CUDA_LAUNCH(calculateVolumes, defaultPolicy,
                r, volPtr, numBubbles, env->getPi());
    CubbleFloatType volume = cubWrapper->reduce<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, volPtr, numBubbles);

    return volume;
}

CubbleFloatType Simulator::getAverageRadius()
{
    CubbleFloatType *r = bubbleData.getRowPtr((size_t)BP::R);
    CubbleFloatType avgRad = cubWrapper->reduce<CubbleFloatType, CubbleFloatType *, CubbleFloatType *>(&cub::DeviceReduce::Sum, r, numBubbles);
    avgRad /= numBubbles;

    return avgRad;
}

void Simulator::getBubbles(std::vector<Bubble> &bubbles) const
{
    bubbles.clear();
    bubbles.resize(numBubbles);

    size_t memoryStride = bubbleData.getWidth();
    CubbleFloatType *devX = bubbleData.getRowPtr((size_t)BP::X);
    std::vector<CubbleFloatType> xyzr;
    xyzr.resize(memoryStride * 4);

    CUDA_CALL(cudaMemcpy(xyzr.data(), devX, sizeof(CubbleFloatType) * 4 * memoryStride, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numBubbles; ++i)
    {
        Bubble b;
        fpvec pos(-1, -1, -1);
        pos.x = xyzr[i];
        pos.y = xyzr[i + memoryStride];
        pos.z = xyzr[i + 2 * memoryStride];
        b.setPos(pos);
        b.setRadius(xyzr[i + 3 * memoryStride]);
        bubbles[i] = b;
    }
}

CubbleFloatType Simulator::getInvRho()
{
    CubbleFloatType invRho = 0;
    CUDA_CALL(cudaMemcpy(static_cast<void *>(&invRho), static_cast<void *>(dir), sizeof(CubbleFloatType), cudaMemcpyDeviceToHost));

    return invRho;
}

void Simulator::transformPositions(bool normalize)
{
    ExecutionPolicy policy;
    policy.gridSize = dim3(256, 1, 1);
    policy.blockSize = dim3(128, 1, 1);
    policy.stream = 0;
    policy.sharedMemBytes = 0;

    CUDA_LAUNCH(transformPositionsKernel, policy,
                normalize, numBubbles, env->getLbb(), env->getTfr(),
                bubbleData.getRowPtr((size_t)BP::X),
                bubbleData.getRowPtr((size_t)BP::Y),
                bubbleData.getRowPtr((size_t)BP::Z));
}
} // namespace cubble
