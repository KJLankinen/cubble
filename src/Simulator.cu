// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
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
    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    double timeStep = env->getTimeStep();

    cudaLaunch(defaultPolicy, resetKernel,
               0.0, numBubbles,
               dxdtOld, dydtOld, dzdtOld, drdtOld);

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    cudaLaunch(pairPolicy, velocityKernel,
               numBubbles, env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
               interval.x, lbb.x, PBC_X == 1, x, dxdtOld,
               interval.y, lbb.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
               ,
               interval.z, lbb.z, PBC_Z == 1, z, dzdtOld
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

    cudaLaunch(pairPolicy, velocityKernel,
               numBubbles, env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
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
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const dvec interval = tfr - lbb;
    const double minRad = env->getMinRad();
    ExecutionPolicy defaultPolicy(128, numBubbles);

    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

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

    double error = 100000;
    size_t numLoopsDone = 0;
    do
    {
        NVTX_RANGE_PUSH_A("Integration step");

        cudaLaunch(defaultPolicy, resetKernel,
                   0.0, numBubbles,
                   dxdtPrd, dydtPrd, dzdtPrd, drdtPrd, freeArea, energies);

        doPrediction(defaultPolicy, timeStep, useGasExchange);
        CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, defaultPolicy.stream));

        cudaLaunch(pairPolicy, velocityKernel,
                   numBubbles, env->getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), rPrd,
                   interval.x, lbb.x, PBC_X == 1, xPrd, dxdtPrd,
                   interval.y, lbb.y, PBC_Y == 1, yPrd, dydtPrd
#if (NUM_DIM == 3)
                   ,
                   interval.z, lbb.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
        );

        if (useGasExchange)
        {
            pairPolicy.stream = gasExchangePolicy.stream;
            CUDA_CALL(cudaStreamWaitEvent(gasExchangePolicy.stream, asyncCopyDHEvent, 0));
            cudaLaunch(pairPolicy, gasExchangeKernel,
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

            cudaLaunch(gasExchangePolicy, freeAreaKernel,
                       numBubbles, env->getPi(), rPrd, freeArea, errors);

            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, errors, dtfapr, numBubbles, gasExchangePolicy.stream);
            cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, freeArea, dtfa, numBubbles, gasExchangePolicy.stream);
            cudaLaunch(gasExchangePolicy, finalRadiusChangeRateKernel,
                       drdtPrd, rPrd, freeArea, numBubbles, 1.0 / env->getPi(), env->getKappa(), env->getKParameter());

            CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, gasExchangePolicy.stream));
            CUDA_CALL(cudaStreamWaitEvent(0, asyncCopyDHEvent, 0));

            pairPolicy.stream = 0;
        }

        doCorrection(defaultPolicy, timeStep, useGasExchange);

        CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, defaultPolicy.stream));
        CUDA_CALL(cudaStreamWaitEvent(asyncCopyDHStream, asyncCopyDHEvent, 0));
        cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, errors, dtfa, numBubbles, asyncCopyDHStream);
        CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(double), cudaMemcpyDeviceToHost, asyncCopyDHStream));
        CUDA_CALL(cudaEventRecord(asyncCopyDHEvent, asyncCopyDHStream));

        CUDA_CALL(cudaEventSynchronize(asyncCopyDHEvent));
        error = pinnedDouble.get()[0];
        if (error < env->getErrorTolerance() && timeStep < 0.1)
            timeStep *= 1.9;
        else if (error > env->getErrorTolerance())
            timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > env->getErrorTolerance());

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    cudaLaunch(defaultPolicy, boundaryWrapKernel,
               numBubbles
#if (PBC_X == 1)
               ,
               xPrd, lbb.x, tfr.x
#endif
#if (PBC_Y == 1)
               ,
               yPrd, lbb.y, tfr.y
#endif
#if (PBC_Z == 1)
               ,
               zPrd, lbb.z, tfr.z
#endif
    );
#endif

    defaultPolicy.stream = asyncCopyDDStream;
    cudaLaunch(defaultPolicy, setFlagIfGreaterThanConstantKernel, numBubbles, aboveMinRadFlags.getRowPtr(0), rPrd, env->getMinRad());
    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles, asyncCopyDDStream);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, asyncCopyDDStream));
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, rPrd, static_cast<double *>(dtfa), numBubbles, asyncCopyDDStream);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double), cudaMemcpyDeviceToHost, asyncCopyDDStream));
    CUDA_CALL(cudaEventRecord(asyncCopyDDEvent, asyncCopyDDStream));

    updateData();

    ++integrationStep;
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    CUDA_CALL(cudaEventSynchronize(asyncCopyDDEvent));
    const int numBubblesAboveMinRad = pinnedInt.get()[0];
    const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;
    if (shouldDeleteBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    if (shouldDeleteBubbles || integrationStep % 50 == 0)
        updateCellsAndNeighbors();

    bool continueSimulation = numBubbles > env->getMinNumBubbles();

#if (NUM_DIM == 3)
    continueSimulation &= pinnedDouble.get()[0] < 0.5 * interval.getMinComponent();
#endif

    return continueSimulation;
}

void Simulator::doPrediction(const ExecutionPolicy &policy, double timeStep, bool useGasExchange)
{
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

    double *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BP::DRDT_OLD);

    if (useGasExchange)
        cudaLaunch(policy, predictKernel,
                   numBubbles, timeStep,
                   xPrd, x, dxdt, dxdtOld,
                   yPrd, y, dydt, dydtOld,
                   zPrd, z, dzdt, dzdtOld,
                   rPrd, r, drdt, drdtOld);
    else
        cudaLaunch(policy, predictKernel,
                   numBubbles, timeStep,
                   xPrd, x, dxdt, dxdtOld,
                   yPrd, y, dydt, dydtOld,
                   zPrd, z, dzdt, dzdtOld);
}

void Simulator::doCorrection(const ExecutionPolicy &policy, double timeStep, bool useGasExchange)
{
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

    double *errors = bubbleData.getRowPtr((size_t)BP::ERROR);

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

    double *x = bubbleData.getRowPtr((size_t)BP::X);
    double *y = bubbleData.getRowPtr((size_t)BP::Y);
    double *z = bubbleData.getRowPtr((size_t)BP::Z);
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    cellData.setBytesToZero();
    bubbleCellIndices.setBytesToZero();

    ExecutionPolicy defaultPolicy = {};
    defaultPolicy.blockSize = dim3(128, 1, 1);
    defaultPolicy.gridSize = dim3(256, 1, 1);
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

    cubWrapper->histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven,
                                                bubbleCellIndices.getRowPtr(2),
                                                sizes,
                                                numCells + 1,
                                                0,
                                                numCells,
                                                numBubbles);

    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
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

    dvec interval = env->getTfr() - env->getLbb();

    ExecutionPolicy findPolicy;
    findPolicy.blockSize = dim3(128, 1, 1);
    findPolicy.gridSize = gridSize;
    findPolicy.sharedMemBytes = 0;

    CUDA_CALL(cudaMemset(np, 0, sizeof(int)));

    for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        findPolicy.stream = neighborStreamVec[i];
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], asyncCopyDDEvent, 0));
        CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], asyncCopyDHEvent, 0));
        cudaLaunch(findPolicy, neighborSearch,
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
    CUDA_CALL(cudaStreamWaitEvent(0, asyncCopyDDEvent, 0));
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
    const int totalNumCells = std::ceil((float)numBubbles / env->getNumBubblesPerCell());
    dvec interval = env->getTfr() - env->getLbb();
    interval /= interval.x;
#if (NUM_DIM == 3)
    float nx = std::cbrt((float)totalNumCells / (interval.y * interval.z));
#else
    float nx = std::sqrt((float)totalNumCells / interval.y);
#endif
    ivec grid = (nx * interval).ceil();
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

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const dvec interval = tfr - lbb;

    cudaLaunch(pairPolicy, potentialEnergyKernel,
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

    ElasticEnergy = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                   bubbleData.getRowPtr((size_t)BP::ENERGY),
                                                                   numBubbles);
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