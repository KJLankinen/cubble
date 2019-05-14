// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
#include "Kernels.cuh"

#include "cub.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <curand.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <vtk-8.0/vtkPoints.h>
#include <vtk-8.0/vtkSmartPointer.h>
#include <vtk-8.0/vtkDoubleArray.h>
#include <vtk-8.0/vtkUnstructuredGrid.h>
#include <vtk-8.0/vtkXMLUnstructuredGridWriter.h>
#include <vtk-8.0/vtkFieldData.h>
#include <vtk-8.0/vtkPointData.h>

namespace cubble
{

bool Simulator::init(const char *inputFileName, const char *outputFileName)
{
    std::ifstream inputFileStream(inputFileName, std::ios::in);
    if (!inputFileStream.is_open())
    {
        std::cerr << "Couldn't open input file!" << std::endl;
        return false;
    }
    nlohmann::json j;
    inputFileStream >> j;
    properties = j;

    dvec relDim = properties.boxRelativeDimensions;
    relDim /= relDim.x;
    const float d = 2 * properties.avgRad;
#if (NUM_DIM == 3)
    const float x = std::cbrt(properties.numBubbles * d * d * d / (relDim.y * relDim.z));
    dvec boxSize = relDim * x;
    const ivec bubblesPerDim(std::ceil(boxSize.x / d), std::ceil(boxSize.y / d), std::ceil(boxSize.z / d));
    numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
#else
    const float x = std::sqrt(properties.numBubbles * d * d / relDim.y);
    dvec boxSize = relDim * x;
    boxSize.z = 0;
    const ivec bubblesPerDim(std::ceil(boxSize.x / d), std::ceil(boxSize.y / d), 0);
    numBubbles = bubblesPerDim.x * bubblesPerDim.y;
#endif
    bubblesPerDimAtStart = bubblesPerDim;
    boxSize = d * bubblesPerDim.asType<double>();
    tfr = boxSize + lbb;
    interval = boxSize;
    simulationBoxVolume = NUM_DIM == 2 ? interval.x * interval.y
                                       : interval.x * interval.y * interval.z;

    cubWrapper = std::make_shared<CubWrapper>(numBubbles * sizeof(double));

    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&deviceData), sizeof(double) * numBubbles * numAliases));
    dataStride = numBubbles;
    adp.x = deviceData;
    adp.y = deviceData + 1 * dataStride;
    adp.z = deviceData + 2 * dataStride;
    adp.r = deviceData + 3 * dataStride;
    adp.dxdt = deviceData + 4 * dataStride;
    adp.dydt = deviceData + 5 * dataStride;
    adp.dzdt = deviceData + 6 * dataStride;
    adp.drdt = deviceData + 7 * dataStride;
    adp.dxdtO = deviceData + 8 * dataStride;
    adp.dydtO = deviceData + 9 * dataStride;
    adp.dzdtO = deviceData + 10 * dataStride;
    adp.drdtO = deviceData + 11 * dataStride;
    adp.x0 = deviceData + 12 * dataStride;
    adp.y0 = deviceData + 13 * dataStride;
    adp.z0 = deviceData + 14 * dataStride;
    adp.s = deviceData + 15 * dataStride;
    adp.d = deviceData + 16 * dataStride;
    adp.xP = deviceData + 17 * dataStride;
    adp.yP = deviceData + 18 * dataStride;
    adp.zP = deviceData + 19 * dataStride;
    adp.rP = deviceData + 20 * dataStride;
    adp.dxdtP = deviceData + 21 * dataStride;
    adp.dydtP = deviceData + 22 * dataStride;
    adp.dzdtP = deviceData + 23 * dataStride;
    adp.drdtP = deviceData + 24 * dataStride;
    adp.error = deviceData + 25 * dataStride;
    adp.dummy1 = deviceData + 26 * dataStride;
    adp.dummy2 = deviceData + 27 * dataStride;
    adp.dummy3 = deviceData + 28 * dataStride;
    adp.dummy4 = deviceData + 29 * dataStride;
    adp.dummy5 = deviceData + 30 * dataStride;
    adp.dummy6 = deviceData + 31 * dataStride;
    adp.dummy7 = deviceData + 32 * dataStride;
    adp.dummy8 = deviceData + 33 * dataStride;

    aboveMinRadFlags = DeviceArray<int>(numBubbles, 2u);
    bubbleCellIndices = DeviceArray<int>(numBubbles, 4u);
    pairs = DeviceArray<int>(8 * numBubbles, 4u);
    wrappedFlags = DeviceArray<bool>(numBubbles, 6);
    CUDA_CALL(cudaMemset(wrappedFlags.get(), 0, wrappedFlags.getSizeInBytes()));

    const dim3 gridSize = getGridSize();
    size_t numCells = gridSize.x * gridSize.y * gridSize.z;
    cellData = DeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);

    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfa), dTotalFreeArea));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfapr), dTotalFreeAreaPerRadius));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&mbpc), dMaxBubblesPerCell));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dvm), dVolumeMultiplier));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtv), dTotalVolume));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&np), dNumPairs));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dir), dInvRho));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dta), dTotalArea));
    CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dasai), dAverageSurfaceAreaIn));

    CUDA_ASSERT(cudaStreamCreateWithFlags(&nonBlockingStream1, cudaStreamNonBlocking));
    CUDA_ASSERT(cudaStreamCreateWithFlags(&nonBlockingStream2, cudaStreamNonBlocking));
    CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent1, cudaEventBlockingSync));
    CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent2, cudaEventBlockingSync));

    for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
    {
        neighborStreamVec.emplace_back();
        neighborEventVec.emplace_back();
        CUDA_ASSERT(cudaStreamCreateWithFlags(&neighborStreamVec[i], cudaStreamNonBlocking));
        CUDA_ASSERT(cudaEventCreate(&neighborEventVec[i]));
    }

    pinnedInt = PinnedHostArray<int>(1);
    pinnedDouble = PinnedHostArray<double>(1);

    printRelevantInfoOfCurrentDevice();

    return true;
}

void Simulator::deinit()
{
    saveSnapshotToFile();

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(static_cast<void *>(deviceData)));

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

void Simulator::run()
{
    std::cout << "======\nSetup\n======" << std::endl;

    setupSimulation();
    saveSnapshotToFile();

    std::cout << "Letting bubbles settle after they've been created and before scaling or stabilization." << std::endl;
    for (size_t i = 0; i < (size_t)properties.numStepsToRelax; ++i)
        integrate();

    saveSnapshotToFile();

    const double phiTarget = properties.phiTarget;
    double bubbleVolume = getVolumeOfBubbles();
    double phi = bubbleVolume / simulationBoxVolume;

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;

    transformPositions(true);
    const dvec relativeSize = properties.boxRelativeDimensions;
    const double temp = bubbleVolume / (NUM_DIM == 2 ? phiTarget * relativeSize.x * relativeSize.y
                                                     : phiTarget * relativeSize.x * relativeSize.y * relativeSize.z);
    const double t = NUM_DIM == 2 ? std::sqrt(temp) : std::cbrt(temp);
    tfr = dvec(t, t, t) * relativeSize;
    transformPositions(false);
    interval = tfr - lbb;
    simulationBoxVolume = NUM_DIM == 2 ? interval.x * interval.y
                                       : interval.x * interval.y * interval.z;

    phi = bubbleVolume / simulationBoxVolume;

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    saveSnapshotToFile();

    std::cout << "=============\nStabilization\n=============" << std::endl;

    integrate();
    calculateEnergy();
    int numSteps = 0;
    const int failsafe = 500;
    double energy2 = elasticEnergy;

    while (true)
    {
        double energy1 = energy2;
        double time = 0;

        for (int i = 0; i < properties.numStepsToRelax; ++i)
        {
            integrate();
            time += properties.timeStep;
        }

        calculateEnergy();
        energy2 = elasticEnergy;
        double deltaEnergy = std::abs(energy2 - energy1) / time;
        deltaEnergy *= 0.5 * properties.sigmaZero;

        if (deltaEnergy < properties.maxDeltaEnergy)
        {
            std::cout << "Final delta energy " << deltaEnergy
                      << " after " << (numSteps + 1) * properties.numStepsToRelax
                      << " steps."
                      << " Energy before: " << energy1
                      << ", energy after: " << energy2
                      << ", time: " << time * properties.kParameter / (properties.avgRad * properties.avgRad)
                      << std::endl;
            break;
        }
        else if (numSteps > failsafe)
        {
            std::cout << "Over " << failsafe * properties.numStepsToRelax
                      << " steps taken and required delta energy not reached."
                      << " Check parameters."
                      << std::endl;
            break;
        }
        else
            std::cout << "Number of simulation steps relaxed: "
                      << (numSteps + 1) * properties.numStepsToRelax
                      << ", delta energy: " << deltaEnergy
                      << ", energy before: " << energy1
                      << ", energy after: " << energy2
                      << std::endl;

        ++numSteps;
    }

    saveSnapshotToFile();

    std::cout << "==========\nSimulation\n==========" << std::endl;

    simulationTime = 0;
    setStartingPositions();

    int timesPrinted = 1;
    bool stopSimulation = false;
    std::stringstream dataStream;

    while (!stopSimulation)
    {
        const double scaledTime = getScaledTime();
        if ((int)scaledTime >= timesPrinted)
        {
            double phi = getVolumeOfBubbles() / simulationBoxVolume;
            double relativeRadius = getAverageProperty(adp.r) / properties.avgRad;
            dataStream << scaledTime
                       << " " << relativeRadius
                       << " " << maxBubbleRadius / properties.avgRad
                       << " " << numBubbles
                       << " " << 1.0 / (getInvRho() * properties.avgRad)
                       << " " << getAverageProperty(adp.d)
                       << " " << getAverageProperty(adp.s)
                       << "\n";

            std::cout << "t*: " << scaledTime
                      << " <R>/<R_in>: " << relativeRadius
                      << " #b: " << numBubbles
                      << " phi: " << phi
                      << std::endl;

            // Only write snapshots when t* is a power of 2.
            if ((timesPrinted & (timesPrinted - 1)) == 0)
                saveSnapshotToFile();

            ++timesPrinted;
        }
    }

    std::ofstream file(properties.dataFilename);
    if (file.is_open())
        file << dataStream.str() << std::endl;
}

void Simulator::setupSimulation()
{
    generateBubbles();

    const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), numBubbles);
    if (numBubblesAboveMinRad < numBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    updateCellsAndNeighbors();

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    double timeStep = properties.timeStep;

    KERNEL_LAUNCH(resetKernel, defaultPolicy,
                  0.0, numBubbles,
                  adp.dxdtO,
                  adp.dydtO,
                  adp.dzdtO,
                  adp.drdtO,
                  adp.d,
                  adp.s);

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    KERNEL_LAUNCH(velocityPairKernel, pairPolicy,
                  properties.fZeroPerMuZero, pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r,
                  interval.x, lbb.x, PBC_X == 1, adp.x, adp.dxdtO,
                  interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO
#if (NUM_DIM == 3)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, adp.z, adp.dzdtO
#endif
    );

    KERNEL_LAUNCH(eulerKernel, defaultPolicy,
                  numBubbles, timeStep,
                  adp.x, adp.dxdtO,
                  adp.y, adp.dydtO
#if (NUM_DIM == 3)
                  ,
                  adp.z, adp.dzdtO
#endif
    );

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    KERNEL_LAUNCH(boundaryWrapKernel, defaultPolicy,
                  numBubbles
#if (PBC_X == 1)
                  ,
                  adp.x, lbb.x, tfr.x, wrappedFlags.getRowPtr(3)
#endif
#if (PBC_Y == 1)
                                           ,
                  adp.y, lbb.y, tfr.y, wrappedFlags.getRowPtr(4)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                                           ,
                  adp.z, lbb.z, tfr.z, wrappedFlags.getRowPtr(5)
#endif
    );
#endif

    KERNEL_LAUNCH(resetKernel, defaultPolicy,
                  0.0, numBubbles,
                  adp.dxdtO, adp.dydtO, adp.dzdtO, adp.drdtO);

    KERNEL_LAUNCH(velocityPairKernel, pairPolicy,
                  properties.fZeroPerMuZero, pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r,
                  interval.x, lbb.x, PBC_X == 1, adp.x, adp.dxdtO,
                  interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO
#if (NUM_DIM == 3)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, adp.z, adp.dzdtO
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

    double error = 100000;
    size_t numLoopsDone = 0;

    do
    {
        NVTX_RANGE_PUSH_A("Integration step");

        doReset(defaultPolicy);
        doPrediction(defaultPolicy, properties.timeStep, useGasExchange, blockingEvent2);
        doVelocity(pairPolicy);
        if (useGasExchange)
            doGasExchange(pairPolicy, blockingEvent2, pairPolicy.stream);
        doCorrection(defaultPolicy, properties.timeStep, useGasExchange, nonBlockingStream2);

        error = doError();
        if (error < properties.errorTolerance && properties.timeStep < 0.1)
            properties.timeStep *= 1.9;
        else if (error > properties.errorTolerance)
            properties.timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > properties.errorTolerance);

    // Holy crap this is ugly. Anyway, don't do the calculations, when stabilizing/equilibrating.
    if (useGasExchange)
    {
        KERNEL_LAUNCH(pathLengthDistanceKernel, defaultPolicy,
                      numBubbles,
                      adp.s,
                      adp.d,
                      adp.xP, adp.x, adp.x0, wrappedFlags.getRowPtr(0), interval.x,
                      adp.yP, adp.y, adp.y0, wrappedFlags.getRowPtr(1), interval.y
#if (NUM_DIM == 3)
                      ,
                      adp.zP, adp.z, adp.z0, wrappedFlags.getRowPtr(2), interval.z
#endif
        );
    }
    doBoundaryWrap(defaultPolicy);
    doBubbleSizeChecks(defaultPolicy, nonBlockingStream1, blockingEvent1);
    updateData();

    ++integrationStep;
    simulationTime += properties.timeStep;

    CUDA_CALL(cudaEventSynchronize(blockingEvent1));

    const int numBubblesAboveMinRad = pinnedInt.get()[0];
    const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;

    if (shouldDeleteBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    if (shouldDeleteBubbles || integrationStep % 50 == 0)
        updateCellsAndNeighbors();

    bool continueSimulation = numBubbles > properties.minNumBubbles;

    maxBubbleRadius = pinnedDouble.get()[0];
#if (NUM_DIM == 3)
    continueSimulation &= maxBubbleRadius < 0.5 * interval.getMinComponent();
#endif

    return continueSimulation;
}

void Simulator::doPrediction(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaEvent_t &eventToMark)
{
    if (useGasExchange)
    {
        KERNEL_LAUNCH(predictKernel, policy,
                      numBubbles, timeStep,
                      adp.xP, adp.x, adp.dxdt, adp.dxdtO,
                      adp.yP, adp.y, adp.dydt, adp.dydtO,
#if (NUM_DIM == 3)
                      adp.zP, adp.z, adp.dzdt, adp.dzdtO,
#endif
                      adp.rP, adp.r, adp.drdt, adp.drdtO);
    }
    else
    {
        KERNEL_LAUNCH(predictKernel, policy,
                      numBubbles, timeStep,
                      adp.xP, adp.x, adp.dxdt, adp.dxdtO,
                      adp.yP, adp.y, adp.dydt, adp.dydtO
#if (NUM_DIM == 3)
                      ,
                      adp.zP, adp.z, adp.dzdt, adp.dzdtO
#endif
        );
    }

    CUDA_CALL(cudaEventRecord(eventToMark, policy.stream));
}

void Simulator::doCorrection(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaStream_t &streamThatShouldWait)
{
    if (useGasExchange)
    {
        KERNEL_LAUNCH(correctKernel, policy,
                      numBubbles, timeStep, adp.error,
                      adp.xP, adp.x, adp.dxdt, adp.dxdtP,
                      adp.yP, adp.y, adp.dydt, adp.dydtP,
#if (NUM_DIM == 3)
                      adp.zP, adp.z, adp.dzdt, adp.dzdtP,
#endif
                      adp.rP, adp.r, adp.drdt, adp.drdtP);
    }
    else
    {
        KERNEL_LAUNCH(correctKernel, policy,
                      numBubbles, timeStep, adp.error,
                      adp.xP, adp.x, adp.dxdt, adp.dxdtP,
                      adp.yP, adp.y, adp.dydt, adp.dydtP
#if (NUM_DIM == 3)
                      ,
                      adp.zP, adp.z, adp.dzdt, adp.dzdtP
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

    KERNEL_LAUNCH(gasExchangeKernel, policy,
                  numBubbles,
                  pi,
                  pairs.getRowPtr(0),
                  pairs.getRowPtr(1),
                  adp.rP,
                  adp.drdtP,
                  adp.dummy1,
                  interval.x, PBC_X == 1, adp.xP,
                  interval.y, PBC_Y == 1, adp.yP
#if (NUM_DIM == 3)
                  ,
                  interval.z, PBC_Z == 1, adp.zP
#endif
    );

    KERNEL_LAUNCH(freeAreaKernel, gasExchangePolicy,
                  numBubbles, pi, adp.rP, adp.dummy1, adp.error, adp.dummy2);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.error, dtfapr, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtfa, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy2, dta, numBubbles, gasExchangePolicy.stream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, gasExchangePolicy,
                  adp.drdtP, adp.rP, adp.dummy1, numBubbles, 1.0 / pi, properties.kappa, properties.kParameter);

    CUDA_CALL(cudaEventRecord(blockingEvent2, gasExchangePolicy.stream));
    CUDA_CALL(cudaStreamWaitEvent(streamThatShouldWait, blockingEvent2, 0));
}

void Simulator::doVelocity(const ExecutionPolicy &policy)
{
    KERNEL_LAUNCH(velocityPairKernel, policy,
                  properties.fZeroPerMuZero, pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP,
                  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP,
                  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#if (NUM_DIM == 3)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
    );

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    KERNEL_LAUNCH(velocityWallKernel, policy,
                  numBubbles, properties.fZeroPerMuZero, pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP
#if (PBC_X == 0)
                  ,
                  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP
#endif
#if (PBC_Y == 0)
                  ,
                  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#endif
#if (NUM_DIM == 3 && PBC_Z == 0)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
    );
#endif

#if USE_FLOW
    int *numNeighbors = bubbleCellIndices.getRowPtr(0);

    KERNEL_LAUNCH(neighborVelocityKernel, policy,
                  pairs.getRowPtr(0), pairs.getRowPtr(1), numNeighbors,
                  adp.dummy1, adp.dxdtO,
                  adp.dummy2, adp.dydtO
#if (NUM_DIM == 3)
                  ,
                  adp.dummy3, adp.dzdtO
#endif
    );

    KERNEL_LAUNCH(flowVelocityKernel, policy,
                  numBubbles, numNeighbors,
                  adp.dummy1, adp.dxdtP,
                  adp.dummy2, adp.dydtP
#if (NUM_DIM == 3)
                  ,
                  adp.dummy3, adp.dzdtP
#endif
    );
#endif
}

void Simulator::doReset(const ExecutionPolicy &policy)
{
    KERNEL_LAUNCH(resetKernel, policy,
                  0.0, numBubbles,
                  adp.dxdtP,
                  adp.dydtP,
                  adp.dzdtP,
                  adp.drdtP,
                  adp.dummy1,
                  adp.dummy2);
}

double Simulator::doError()
{
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, adp.error, dtfa, numBubbles, nonBlockingStream2);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(double), cudaMemcpyDeviceToHost, nonBlockingStream2));
    CUDA_CALL(cudaEventRecord(blockingEvent2, nonBlockingStream2));
    CUDA_CALL(cudaEventSynchronize(blockingEvent2));

    return pinnedDouble.get()[0];
}

void Simulator::doBoundaryWrap(const ExecutionPolicy &policy)
{
#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    KERNEL_LAUNCH(boundaryWrapKernel, policy,
                  numBubbles
#if (PBC_X == 1)
                  ,
                  adp.xP, lbb.x, tfr.x, wrappedFlags.getRowPtr(0)
#endif
#if (PBC_Y == 1)
                                            ,
                  adp.yP, lbb.y, tfr.y, wrappedFlags.getRowPtr(1)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                                            ,
                  adp.zP, lbb.z, tfr.z, wrappedFlags.getRowPtr(2)
#endif
    );
#endif
}

void Simulator::doBubbleSizeChecks(ExecutionPolicy policy, cudaStream_t &streamToUse, cudaEvent_t &eventToMark)
{
    policy.stream = streamToUse;

    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, policy,
                  numBubbles,
                  aboveMinRadFlags.getRowPtr(0),
                  adp.rP,
                  properties.minRad);

    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, streamToUse));

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, adp.rP, static_cast<double *>(dtfa), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double), cudaMemcpyDeviceToHost, streamToUse));

    CUDA_CALL(cudaEventRecord(eventToMark, streamToUse));
}

void Simulator::generateBubbles()
{
    std::cout << "Starting to generate data for bubbles." << std::endl;

    const int rngSeed = properties.rngSeed;
    const double avgRad = properties.avgRad;
    const double stdDevRad = properties.stdDevRad;

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));

    CURAND_CALL(curandGenerateUniformDouble(generator, adp.x, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, adp.y, numBubbles));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, adp.z, numBubbles));
#endif
    CURAND_CALL(curandGenerateUniformDouble(generator, adp.rP, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, adp.r, numBubbles, avgRad, stdDevRad));

    CURAND_CALL(curandDestroyGenerator(generator));

    ExecutionPolicy defaultPolicy(128, numBubbles);
    assert(bubblesPerDimAtStart.x > 0);
    assert(bubblesPerDimAtStart.y > 0);
#if (NUM_DIM == 3)
    assert(bubblesPerDimAtStart.z > 0);
#endif
    KERNEL_LAUNCH(assignDataToBubbles, defaultPolicy,
                  adp.x, adp.y, adp.z, adp.xP, adp.yP, adp.zP, adp.r, adp.rP,
                  aboveMinRadFlags.getRowPtr(0), bubblesPerDimAtStart,
                  tfr, lbb, avgRad, properties.minRad, pi, numBubbles);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.rP, dasai, numBubbles, defaultPolicy.stream);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.rP), static_cast<void *>(adp.r),
                              sizeof(double) * dataStride,
                              cudaMemcpyDeviceToDevice,
                              defaultPolicy.stream));
}

void Simulator::updateCellsAndNeighbors()
{
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    cellData.setBytesToZero();
    bubbleCellIndices.setBytesToZero();

    ExecutionPolicy defaultPolicy = {};
    defaultPolicy.blockSize = dim3(128, 1, 1);
    defaultPolicy.gridSize = dim3(256, 1, 1);
    ExecutionPolicy asyncCopyDDPolicy(128, numBubbles, 0, nonBlockingStream1);
    KERNEL_LAUNCH(assignBubblesToCells, defaultPolicy,
                  adp.x, adp.y, adp.z,
                  bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3),
                  lbb, tfr, cellDim, numBubbles);

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

    KERNEL_LAUNCH(reorganizeKernel, asyncCopyDDPolicy,
                  numBubbles, ReorganizeType::COPY_FROM_INDEX, bubbleIndices, bubbleIndices,
                  adp.x, adp.xP,
                  adp.y, adp.yP,
                  adp.z, adp.zP,
                  adp.r, adp.rP,
                  adp.dxdt, adp.dxdtP,
                  adp.dydt, adp.dydtP,
                  adp.dzdt, adp.dzdtP,
                  adp.drdt, adp.drdtP,
                  adp.dxdtO, adp.error,
                  adp.dydtO, adp.dummy1,
                  adp.dzdtO, adp.dummy2,
                  adp.drdtO, adp.dummy3,
                  adp.x0, adp.dummy4,
                  adp.y0, adp.dummy5,
                  adp.z0, adp.dummy6,
                  adp.s, adp.dummy7,
                  adp.d, adp.dummy8,
                  wrappedFlags.getRowPtr(0), wrappedFlags.getRowPtr(3),
                  wrappedFlags.getRowPtr(1), wrappedFlags.getRowPtr(4),
                  wrappedFlags.getRowPtr(2), wrappedFlags.getRowPtr(5));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x),
                              static_cast<void *>(adp.xP),
                              sizeof(double) * numAliases / 2 * dataStride,
                              cudaMemcpyDeviceToDevice,
                              nonBlockingStream1));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrappedFlags.getRowPtr(0)),
                              static_cast<void *>(wrappedFlags.getRowPtr(3)),
                              wrappedFlags.getSizeInBytes() / 2,
                              cudaMemcpyDeviceToDevice,
                              nonBlockingStream1));

    CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream1));

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
        KERNEL_LAUNCH(neighborSearch, findPolicy,
                      i, numBubbles, numCells, static_cast<int>(pairs.getWidth()),
                      offsets, sizes, pairs.getRowPtr(2), pairs.getRowPtr(3), adp.r,
                      interval.x, PBC_X == 1, adp.x,
                      interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
                      ,
                      interval.z, PBC_Z == 1, adp.z
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
    const size_t numBytesToCopy = 4 * sizeof(double) * dataStride;

    CUDA_CALL(cudaMemcpyAsync(adp.dxdtO, adp.dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpyAsync(adp.x, adp.xP, 2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
}

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
    NVTX_RANGE_PUSH_A("BubbleRemoval");
    ExecutionPolicy defaultPolicy(128, numBubbles);

    int *flag = aboveMinRadFlags.getRowPtr(0);

    CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(double)));
    KERNEL_LAUNCH(calculateRedistributedGasVolume, defaultPolicy,
                  adp.dummy1, adp.r, flag, pi, numBubbles);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtv, numBubbles);

    int *newIdx = aboveMinRadFlags.getRowPtr(1);
    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

    KERNEL_LAUNCH(reorganizeKernel, defaultPolicy,
                  numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, flag,
                  adp.x, adp.xP,
                  adp.y, adp.yP,
                  adp.z, adp.zP,
                  adp.r, adp.rP,
                  adp.dxdt, adp.dxdtP,
                  adp.dydt, adp.dydtP,
                  adp.dzdt, adp.dzdtP,
                  adp.drdt, adp.drdtP,
                  adp.dxdtO, adp.error,
                  adp.dydtO, adp.dummy1,
                  adp.dzdtO, adp.dummy2,
                  adp.drdtO, adp.dummy3,
                  adp.x0, adp.dummy4,
                  adp.y0, adp.dummy5,
                  adp.z0, adp.dummy6,
                  adp.s, adp.dummy7,
                  adp.d, adp.dummy8,
                  wrappedFlags.getRowPtr(0), wrappedFlags.getRowPtr(3),
                  wrappedFlags.getRowPtr(1), wrappedFlags.getRowPtr(4),
                  wrappedFlags.getRowPtr(2), wrappedFlags.getRowPtr(5));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x),
                              static_cast<void *>(adp.xP),
                              sizeof(double) * numAliases / 2 * dataStride,
                              cudaMemcpyDeviceToDevice));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrappedFlags.getRowPtr(0)),
                              static_cast<void *>(wrappedFlags.getRowPtr(3)),
                              wrappedFlags.getSizeInBytes() / 2,
                              cudaMemcpyDeviceToDevice));

    numBubbles = numBubblesAboveMinRad;
    KERNEL_LAUNCH(addVolume, defaultPolicy, adp.r, numBubbles);

    NVTX_RANGE_POP();
}

dim3 Simulator::getGridSize()
{
    const int totalNumCells = std::ceil((float)numBubbles / properties.numBubblesPerCell);
    dvec normalizedInterval = interval;
    normalizedInterval /= normalizedInterval.x;
#if (NUM_DIM == 3)
    float nx = std::cbrt((float)totalNumCells / (normalizedInterval.y * normalizedInterval.z));
#else
    float nx = std::sqrt((float)totalNumCells / normalizedInterval.y);
    normalizedInterval.z = 0;
#endif
    ivec grid = (nx * normalizedInterval).floor() + 1;
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

    KERNEL_LAUNCH(potentialEnergyKernel, pairPolicy,
                  numBubbles,
                  pairs.getRowPtr(0),
                  pairs.getRowPtr(1),
                  adp.r,
                  adp.dummy1,
                  interval.x, PBC_X == 1, adp.x,
                  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
                  ,
                  interval.z, PBC_Z == 1, adp.z
#endif
    );

    elasticEnergy = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                   adp.dummy1,
                                                                   numBubbles);
}

double Simulator::getVolumeOfBubbles()
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    KERNEL_LAUNCH(calculateVolumes, defaultPolicy,
                  adp.r, adp.dummy1, numBubbles, pi);
    double volume = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, numBubbles);

    return volume;
}

double Simulator::getInvRho()
{
    double invRho = 0;
    CUDA_CALL(cudaMemcpy(static_cast<void *>(&invRho), static_cast<void *>(dir), sizeof(double), cudaMemcpyDeviceToHost));

    return invRho;
}

void Simulator::transformPositions(bool normalize)
{
    ExecutionPolicy policy;
    policy.gridSize = dim3(256, 1, 1);
    policy.blockSize = dim3(128, 1, 1);
    policy.stream = 0;
    policy.sharedMemBytes = 0;

    KERNEL_LAUNCH(transformPositionsKernel, policy,
                  normalize, numBubbles, lbb, tfr,
                  adp.x,
                  adp.y,
                  adp.z);
}

double Simulator::getAverageProperty(double *p)
{
    return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, p, numBubbles) / numBubbles;
}

void Simulator::setStartingPositions()
{
    const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
    CUDA_CALL(cudaMemcpy(adp.x0, adp.x, numBytesToCopy, cudaMemcpyDeviceToDevice));
}

void Simulator::saveSnapshotToFile()
{
    auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    auto dataSet = vtkSmartPointer<vtkUnstructuredGrid>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto timeArray = vtkSmartPointer<vtkDoubleArray>::New();
    auto radiiArray = vtkSmartPointer<vtkDoubleArray>::New();
    auto velArray = vtkSmartPointer<vtkDoubleArray>::New();

    // Filename
    std::stringstream ss;
    ss << properties.snapshotFilename << "." << writer->GetDefaultFileExtension() << "." << numSnapshots;
    writer->SetFileName((ss.str()).c_str());

    // Time stamp
    timeArray->SetNumberOfTuples(1);
    timeArray->SetTuple1(0, getScaledTime());
    timeArray->SetName("Time");
    dataSet->GetFieldData()->AddArray(timeArray);

    // Points
    const size_t numComp = 7;
    hostData.clear();
    hostData.resize(dataStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), deviceData, sizeof(double) * numComp * dataStride, cudaMemcpyDeviceToHost));
    points->SetNumberOfPoints(numBubbles);

    radiiArray->SetNumberOfComponents(1);
    radiiArray->SetNumberOfTuples(points->GetNumberOfPoints());
    radiiArray->SetName("Radius");

    velArray->SetNumberOfComponents(3);
    velArray->SetNumberOfTuples(points->GetNumberOfPoints());
    velArray->SetName("Velocity");

    std::vector<double> t;
    t.resize(3);

    for (size_t i = 0; i < (size_t)points->GetNumberOfPoints(); ++i)
    {
        t[0] = hostData[i + 0 * dataStride];
        t[1] = hostData[i + 1 * dataStride];
        t[2] = hostData[i + 2 * dataStride];
        points->SetPoint(i, t.data());

        radiiArray->InsertValue(i, hostData[i + 3 * dataStride]);

        t[0] = hostData[i + 4 * dataStride];
        t[1] = hostData[i + 5 * dataStride];
        t[2] = hostData[i + 6 * dataStride];
        velArray->InsertTuple(i, t.data());
    }

    dataSet->GetPointData()->AddArray(radiiArray);
    dataSet->GetPointData()->AddArray(velArray);
    dataSet->SetPoints(points);

    // Remove unused memory
    dataSet->Squeeze();

    // Write
    writer->SetInputData(dataSet);
    writer->SetDataModeToBinary();
    writer->Write();

    ++numSnapshots;
}

} // namespace cubble
