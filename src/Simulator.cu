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
typedef BubbleProperty BP;

bool Simulator::init(const char *inputFileName, const char *outputFileName)
{
    properties = Env(inputFileName, outputFileName);
    properties.readParameters();

    dvec relDim = properties.getBoxRelativeDimensions();
    relDim /= relDim.x;
    const float d = 2 * properties.getAvgRad();
#if (NUM_DIM == 3)
    const float x = std::cbrt(properties.getNumBubbles() * d * d * d / (relDim.y * relDim.z));
    dvec tfr = relDim * x;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), std::ceil(tfr.z / d));
    numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
#else
    const float x = std::sqrt(properties.getNumBubbles() * d * d / relDim.y);
    dvec tfr = relDim * x;
    tfr.z = 0;
    const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), 0);
    numBubbles = bubblesPerDim.x * bubblesPerDim.y;
#endif
    bubblesPerDimAtStart = bubblesPerDim;
    tfr = d * bubblesPerDim.asType<double>();
    properties.setTfr(tfr + properties.getLbb());

    cubWrapper = std::make_shared<CubWrapper>(numBubbles * sizeof(double));
    bubbleData = DeviceArray<double>(numBubbles, (size_t)BP::NUM_VALUES);
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
    properties.writeParameters();

    CUDA_CALL(cudaDeviceSynchronize());

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
    for (size_t i = 0; i < (size_t)properties.getNumStepsToRelax(); ++i)
        integrate();

    saveSnapshotToFile();

    const double phiTarget = properties.getPhiTarget();
    double bubbleVolume = getVolumeOfBubbles();
    double phi = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;

    transformPositions(true);
    const dvec relativeSize = properties.getBoxRelativeDimensions();
#if (NUM_DIM == 3)
    const double t = std::cbrt(getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z));
#else
    const double t = std::sqrt(getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y));
#endif
    properties.setTfr(dvec(t, t, t) * relativeSize);
    transformPositions(false);

    phi = bubbleVolume / properties.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    saveSnapshotToFile();

    std::cout << "=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    integrate();
    calculateEnergy();
    double energy2 = elasticEnergy;

    while (true)
    {
        double energy1 = energy2;
        double time = 0;

        for (int i = 0; i < properties.getNumStepsToRelax(); ++i)
        {
            integrate();
            time += properties.getTimeStep();
        }

        calculateEnergy();
        energy2 = elasticEnergy;
        double deltaEnergy = std::abs(energy2 - energy1) / time;
        deltaEnergy *= 0.5 * properties.getSigmaZero();

        if (deltaEnergy < properties.getMaxDeltaEnergy())
        {
            std::cout << "Final delta energy " << deltaEnergy
                      << " after " << (numSteps + 1) * properties.getNumStepsToRelax()
                      << " steps."
                      << " Energy before: " << energy1
                      << ", energy after: " << energy2
                      << ", time: " << time * properties.getKParameter() / (properties.getAvgRad() * properties.getAvgRad())
                      << std::endl;
            break;
        }
        else if (numSteps > failsafe)
        {
            std::cout << "Over " << failsafe * properties.getNumStepsToRelax()
                      << " steps taken and required delta energy not reached."
                      << " Check parameters."
                      << std::endl;
            break;
        }
        else
            std::cout << "Number of simulation steps relaxed: "
                      << (numSteps + 1) * properties.getNumStepsToRelax()
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

    numSteps = 0;
    int timesPrinted = 1;
    bool stopSimulation = false;
    std::stringstream dataStream;

    while (!stopSimulation)
    {
        if (numSteps == 2000)
        {
            CUDA_PROFILER_START();
        }

        stopSimulation = !integrate(true);

        if (numSteps == 2050)
        {
            CUDA_PROFILER_STOP();
#if (USE_PROFILING == 1)
            break;
#endif
        }

        const double scaledTime = simulationTime * properties.getKParameter() / (properties.getAvgRad() * properties.getAvgRad());
        if ((int)scaledTime >= timesPrinted)
        {
            double phi = getVolumeOfBubbles() / properties.getSimulationBoxVolume();
            double relativeRadius = getAverageProperty(BubbleProperty::R) / properties.getAvgRad();
            dataStream << scaledTime
                       << " " << relativeRadius
                       << " " << getMaxBubbleRadius() / properties.getAvgRad()
                       << " " << numBubbles
                       << " " << 1.0 / (getInvRho() * properties.getAvgRad())
                       << " " << getAverageProperty(BubbleProperty::SQUARED_DISTANCE)
                       << " " << getAverageProperty(BubbleProperty::PATH_LENGTH)
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

        ++numSteps;
    }

    std::ofstream file(properties.getDataFilename());
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

    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();
    const dvec interval = tfr - lbb;
    ExecutionPolicy defaultPolicy(128, numBubbles);
    ExecutionPolicy pairPolicy;
    pairPolicy.blockSize = dim3(128, 1, 1);
    pairPolicy.stream = 0;
    pairPolicy.gridSize = dim3(256, 1, 1);
    pairPolicy.sharedMemBytes = 0;

    double timeStep = properties.getTimeStep();

    KERNEL_LAUNCH(resetKernel, defaultPolicy,
                  0.0, numBubbles,
                  dxdtOld,
                  dydtOld,
                  dzdtOld,
                  drdtOld,
                  bubbleData.getRowPtr((size_t)BP::SQUARED_DISTANCE),
                  bubbleData.getRowPtr((size_t)BP::PATH_LENGTH));

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    KERNEL_LAUNCH(velocityPairKernel, pairPolicy,
                  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
                  interval.x, lbb.x, PBC_X == 1, x, dxdtOld,
                  interval.y, lbb.y, PBC_Y == 1, y, dydtOld
#if (NUM_DIM == 3)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, z, dzdtOld
#endif
    );

    KERNEL_LAUNCH(eulerKernel, defaultPolicy,
                  numBubbles, timeStep,
                  x, dxdtOld,
                  y, dydtOld
#if (NUM_DIM == 3)
                  ,
                  z, dzdtOld
#endif
    );

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    KERNEL_LAUNCH(boundaryWrapKernel, defaultPolicy,
                  numBubbles
#if (PBC_X == 1)
                  ,
                  x, lbb.x, tfr.x, wrappedFlags.getRowPtr(3)
#endif
#if (PBC_Y == 1)
                                       ,
                  y, lbb.y, tfr.y, wrappedFlags.getRowPtr(4)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                                       ,
                  z, lbb.z, tfr.z, wrappedFlags.getRowPtr(5)
#endif
    );
#endif

    KERNEL_LAUNCH(resetKernel, defaultPolicy,
                  0.0, numBubbles,
                  dxdtOld, dydtOld, dzdtOld, drdtOld);

    KERNEL_LAUNCH(velocityPairKernel, pairPolicy,
                  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), r,
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

    double timeStep = properties.getTimeStep();
    double error = 100000;
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
        if (error < properties.getErrorTolerance() && timeStep < 0.1)
            timeStep *= 1.9;
        else if (error > properties.getErrorTolerance())
            timeStep *= 0.5;

        ++numLoopsDone;

        NVTX_RANGE_POP();
    } while (error > properties.getErrorTolerance());

    // Holy crap this is ugly. Anyway, don't do the calculations, when stabilizing/equilibrating.
    if (useGasExchange)
    {
        const dvec interval = properties.getTfr() - properties.getLbb();
        KERNEL_LAUNCH(pathLengthDistanceKernel, defaultPolicy,
                      numBubbles,
                      bubbleData.getRowPtr((size_t)BP::PATH_LENGTH),
                      bubbleData.getRowPtr((size_t)BP::SQUARED_DISTANCE),
                      bubbleData.getRowPtr((size_t)BP::X_PRD), bubbleData.getRowPtr((size_t)BP::X), bubbleData.getRowPtr((size_t)BP::X_START), wrappedFlags.getRowPtr(0), interval.x,
                      bubbleData.getRowPtr((size_t)BP::Y_PRD), bubbleData.getRowPtr((size_t)BP::Y), bubbleData.getRowPtr((size_t)BP::Y_START), wrappedFlags.getRowPtr(1), interval.y
#if (NUM_DIM == 3)
                      ,
                      bubbleData.getRowPtr((size_t)BP::Z_PRD), bubbleData.getRowPtr((size_t)BP::Z), bubbleData.getRowPtr((size_t)BP::Z_START), wrappedFlags.getRowPtr(2), interval.z
#endif
        );
    }
    doBoundaryWrap(defaultPolicy);
    doBubbleSizeChecks(defaultPolicy, nonBlockingStream1, blockingEvent1);
    updateData();

    ++integrationStep;
    properties.setTimeStep(timeStep);
    simulationTime += timeStep;

    CUDA_CALL(cudaEventSynchronize(blockingEvent1));

    const int numBubblesAboveMinRad = pinnedInt.get()[0];
    const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;

    if (shouldDeleteBubbles)
        deleteSmallBubbles(numBubblesAboveMinRad);

    if (shouldDeleteBubbles || integrationStep % 50 == 0)
        updateCellsAndNeighbors();

    bool continueSimulation = numBubbles > properties.getMinNumBubbles();

    maxBubbleRadius = pinnedDouble.get()[0];
#if (NUM_DIM == 3)
    continueSimulation &= maxBubbleRadius < 0.5 * (properties.getTfr() - properties.getLbb()).getMinComponent();
#endif

    return continueSimulation;
}

void Simulator::doPrediction(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaEvent_t &eventToMark)
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
    {
        KERNEL_LAUNCH(predictKernel, policy,
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
        KERNEL_LAUNCH(predictKernel, policy,
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

void Simulator::doCorrection(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaStream_t &streamThatShouldWait)
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
    {
        KERNEL_LAUNCH(correctKernel, policy,
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
        KERNEL_LAUNCH(correctKernel, policy,
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

    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    double *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    double *drdtPrd = bubbleData.getRowPtr((size_t)BP::DRDT_PRD);
    double *errors = bubbleData.getRowPtr((size_t)BP::ERROR);
    double *freeArea = bubbleData.getRowPtr((size_t)BP::FREE_AREA);
    double *volume = bubbleData.getRowPtr((size_t)BP::VOLUME);

    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();
    const dvec interval = tfr - lbb;

    KERNEL_LAUNCH(gasExchangeKernel, policy,
                  numBubbles,
                  properties.getPi(),
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

    KERNEL_LAUNCH(freeAreaKernel, gasExchangePolicy,
                  numBubbles, properties.getPi(), rPrd, freeArea, errors, volume);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, errors, dtfapr, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, freeArea, dtfa, numBubbles, gasExchangePolicy.stream);
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, volume, dta, numBubbles, gasExchangePolicy.stream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, gasExchangePolicy,
                  drdtPrd, rPrd, freeArea, numBubbles, 1.0 / properties.getPi(), properties.getKappa(), properties.getKParameter());

    CUDA_CALL(cudaEventRecord(blockingEvent2, gasExchangePolicy.stream));
    CUDA_CALL(cudaStreamWaitEvent(streamThatShouldWait, blockingEvent2, 0));
}

void Simulator::doVelocity(const ExecutionPolicy &policy)
{
    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();
    const dvec interval = tfr - lbb;

    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);
    double *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    double *dxdtPrd = bubbleData.getRowPtr((size_t)BP::DXDT_PRD);
    double *dydtPrd = bubbleData.getRowPtr((size_t)BP::DYDT_PRD);
    double *dzdtPrd = bubbleData.getRowPtr((size_t)BP::DZDT_PRD);

    KERNEL_LAUNCH(velocityPairKernel, policy,
                  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), rPrd,
                  interval.x, lbb.x, PBC_X == 1, xPrd, dxdtPrd,
                  interval.y, lbb.y, PBC_Y == 1, yPrd, dydtPrd
#if (NUM_DIM == 3)
                  ,
                  interval.z, lbb.z, PBC_Z == 1, zPrd, dzdtPrd
#endif
    );

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    KERNEL_LAUNCH(velocityWallKernel, policy,
                  numBubbles, properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), rPrd
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

#if USE_FLOW
    int *numNeighbors = bubbleCellIndices.getRowPtr(0);
    double *dxdtOld = bubbleData.getRowPtr((size_t)BP::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BP::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BP::DZDT_OLD);
    double *flowVelocityX = bubbleData.getRowPtr((size_t)BP::ENERGY);
    double *flowVelocityY = bubbleData.getRowPtr((size_t)BP::FREE_AREA);
    double *flowVelocityZ = bubbleData.getRowPtr((size_t)BP::ERROR);

    KERNEL_LAUNCH(neighborVelocityKernel, policy,
                  pairs.getRowPtr(0), pairs.getRowPtr(1), numNeighbors,
                  flowVelocityX, dxdtOld,
                  flowVelocityY, dydtOld
#if (NUM_DIM == 3)
                  ,
                  flowVelocityZ, dzdtOld
#endif
    );

    KERNEL_LAUNCH(flowVelocityKernel, policy,
                  numBubbles, numNeighbors,
                  flowVelocityX, dxdtPrd,
                  flowVelocityY, dydtPrd
#if (NUM_DIM == 3)
                  ,
                  flowVelocityZ, dzdtPrd
#endif
    );
#endif
}

void Simulator::doReset(const ExecutionPolicy &policy)
{
    KERNEL_LAUNCH(resetKernel, policy,
                  0.0, numBubbles,
                  bubbleData.getRowPtr((size_t)BP::DXDT_PRD),
                  bubbleData.getRowPtr((size_t)BP::DYDT_PRD),
                  bubbleData.getRowPtr((size_t)BP::DZDT_PRD),
                  bubbleData.getRowPtr((size_t)BP::DRDT_PRD),
                  bubbleData.getRowPtr((size_t)BP::FREE_AREA),
                  bubbleData.getRowPtr((size_t)BP::ENERGY));
}

double Simulator::doError()
{
    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, bubbleData.getRowPtr((size_t)BP::ERROR), dtfa, numBubbles, nonBlockingStream2);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(double), cudaMemcpyDeviceToHost, nonBlockingStream2));
    CUDA_CALL(cudaEventRecord(blockingEvent2, nonBlockingStream2));
    CUDA_CALL(cudaEventSynchronize(blockingEvent2));

    return pinnedDouble.get()[0];
}

void Simulator::doBoundaryWrap(const ExecutionPolicy &policy)
{
    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();

    double *xPrd = bubbleData.getRowPtr((size_t)BP::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BP::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BP::Z_PRD);

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    KERNEL_LAUNCH(boundaryWrapKernel, policy,
                  numBubbles
#if (PBC_X == 1)
                  ,
                  xPrd, lbb.x, tfr.x, wrappedFlags.getRowPtr(0)
#endif
#if (PBC_Y == 1)
                                          ,
                  yPrd, lbb.y, tfr.y, wrappedFlags.getRowPtr(1)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
                                          ,
                  zPrd, lbb.z, tfr.z, wrappedFlags.getRowPtr(2)
#endif
    );
#endif
}

void Simulator::doBubbleSizeChecks(ExecutionPolicy policy, cudaStream_t &streamToUse, cudaEvent_t &eventToMark)
{
    double *rPrd = bubbleData.getRowPtr((size_t)BP::R_PRD);
    policy.stream = streamToUse;

    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, policy,
                  numBubbles,
                  aboveMinRadFlags.getRowPtr(0),
                  rPrd,
                  properties.getMinRad());

    cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, streamToUse));

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, rPrd, static_cast<double *>(dtfa), numBubbles, streamToUse);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double), cudaMemcpyDeviceToHost, streamToUse));

    CUDA_CALL(cudaEventRecord(eventToMark, streamToUse));
}

void Simulator::generateBubbles()
{
    std::cout << "Starting to generate data for bubbles." << std::endl;

    const int rngSeed = properties.getRngSeed();
    const double avgRad = properties.getAvgRad();
    const double stdDevRad = properties.getStdDevRad();
    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();

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
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, z, numBubbles));
#endif
    CURAND_CALL(curandGenerateUniformDouble(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, r, numBubbles, avgRad, stdDevRad));

    CURAND_CALL(curandDestroyGenerator(generator));

    ExecutionPolicy defaultPolicy(128, numBubbles);
    assert(bubblesPerDimAtStart.x > 0);
    assert(bubblesPerDimAtStart.y > 0);
#if (NUM_DIM == 3)
    assert(bubblesPerDimAtStart.z > 0);
#endif
    KERNEL_LAUNCH(assignDataToBubbles, defaultPolicy,
                  x, y, z, xPrd, yPrd, zPrd, r, w,
                  aboveMinRadFlags.getRowPtr(0), bubblesPerDimAtStart,
                  tfr, lbb, avgRad, properties.getMinRad(), properties.getPi(), numBubbles);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, w, dasai, numBubbles, defaultPolicy.stream);
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(w), static_cast<void *>(r),
                              sizeof(double) * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice,
                              defaultPolicy.stream));
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
    ExecutionPolicy asyncCopyDDPolicy(128, numBubbles, 0, nonBlockingStream1);
    KERNEL_LAUNCH(assignBubblesToCells, defaultPolicy,
                  x, y, z, bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3), properties.getLbb(), properties.getTfr(), cellDim, numBubbles);

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
                  bubbleData.getRowPtr((size_t)BP::DRDT_OLD), bubbleData.getRowPtr((size_t)BP::VOLUME),
                  bubbleData.getRowPtr((size_t)BP::X_START), bubbleData.getRowPtr((size_t)BP::TEMP1),
                  bubbleData.getRowPtr((size_t)BP::Y_START), bubbleData.getRowPtr((size_t)BP::TEMP2),
                  bubbleData.getRowPtr((size_t)BP::Z_START), bubbleData.getRowPtr((size_t)BP::TEMP3),
                  bubbleData.getRowPtr((size_t)BP::PATH_LENGTH), bubbleData.getRowPtr((size_t)BP::TEMP4),
                  bubbleData.getRowPtr((size_t)BP::SQUARED_DISTANCE), bubbleData.getRowPtr((size_t)BP::TEMP5),
                  wrappedFlags.getRowPtr(0), wrappedFlags.getRowPtr(3),
                  wrappedFlags.getRowPtr(1), wrappedFlags.getRowPtr(4),
                  wrappedFlags.getRowPtr(2), wrappedFlags.getRowPtr(5));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(x),
                              static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)),
                              sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice,
                              nonBlockingStream1));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrappedFlags.getRowPtr(0)),
                              static_cast<void *>(wrappedFlags.getRowPtr(3)),
                              wrappedFlags.getSizeInBytes() / 2,
                              cudaMemcpyDeviceToDevice,
                              nonBlockingStream1));

    CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream1));

    dvec interval = properties.getTfr() - properties.getLbb();

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
    KERNEL_LAUNCH(calculateRedistributedGasVolume, defaultPolicy,
                  volumes, r, flag, properties.getPi(), numBubbles);

    cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, volumes, dtv, numBubbles);

    int *newIdx = aboveMinRadFlags.getRowPtr(1);
    cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

    KERNEL_LAUNCH(reorganizeKernel, defaultPolicy,
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
                  bubbleData.getRowPtr((size_t)BP::DRDT_OLD), bubbleData.getRowPtr((size_t)BP::VOLUME),
                  bubbleData.getRowPtr((size_t)BP::X_START), bubbleData.getRowPtr((size_t)BP::TEMP1),
                  bubbleData.getRowPtr((size_t)BP::Y_START), bubbleData.getRowPtr((size_t)BP::TEMP2),
                  bubbleData.getRowPtr((size_t)BP::Z_START), bubbleData.getRowPtr((size_t)BP::TEMP3),
                  bubbleData.getRowPtr((size_t)BP::PATH_LENGTH), bubbleData.getRowPtr((size_t)BP::TEMP4),
                  bubbleData.getRowPtr((size_t)BP::SQUARED_DISTANCE), bubbleData.getRowPtr((size_t)BP::TEMP5),
                  wrappedFlags.getRowPtr(0), wrappedFlags.getRowPtr(3),
                  wrappedFlags.getRowPtr(1), wrappedFlags.getRowPtr(4),
                  wrappedFlags.getRowPtr(2), wrappedFlags.getRowPtr(5));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X)),
                              static_cast<void *>(bubbleData.getRowPtr((size_t)BP::X_PRD)),
                              sizeof(double) * (size_t)BP::X_PRD * bubbleData.getWidth(),
                              cudaMemcpyDeviceToDevice));

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrappedFlags.getRowPtr(0)),
                              static_cast<void *>(wrappedFlags.getRowPtr(3)),
                              wrappedFlags.getSizeInBytes() / 2,
                              cudaMemcpyDeviceToDevice));

    numBubbles = numBubblesAboveMinRad;
    KERNEL_LAUNCH(addVolume, defaultPolicy, r, numBubbles);

    NVTX_RANGE_POP();
}

dim3 Simulator::getGridSize()
{
    const int totalNumCells = std::ceil((float)numBubbles / properties.getNumBubblesPerCell());
    dvec interval = properties.getTfr() - properties.getLbb();
    interval /= interval.x;
#if (NUM_DIM == 3)
    float nx = std::cbrt((float)totalNumCells / (interval.y * interval.z));
#else
    float nx = std::sqrt((float)totalNumCells / interval.y);
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

    const dvec tfr = properties.getTfr();
    const dvec lbb = properties.getLbb();
    const dvec interval = tfr - lbb;

    KERNEL_LAUNCH(potentialEnergyKernel, pairPolicy,
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

    elasticEnergy = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum,
                                                                   bubbleData.getRowPtr((size_t)BP::ENERGY),
                                                                   numBubbles);
}

double Simulator::getVolumeOfBubbles()
{
    ExecutionPolicy defaultPolicy(128, numBubbles);
    double *r = bubbleData.getRowPtr((size_t)BP::R);
    double *volPtr = bubbleData.getRowPtr((size_t)BP::VOLUME);
    KERNEL_LAUNCH(calculateVolumes, defaultPolicy,
                  r, volPtr, numBubbles, properties.getPi());
    double volume = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, volPtr, numBubbles);

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
                  normalize, numBubbles, properties.getLbb(), properties.getTfr(),
                  bubbleData.getRowPtr((size_t)BP::X),
                  bubbleData.getRowPtr((size_t)BP::Y),
                  bubbleData.getRowPtr((size_t)BP::Z));
}

double Simulator::getAverageProperty(BubbleProperty property)
{
    return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, bubbleData.getRowPtr((size_t)property), numBubbles) / numBubbles;
}

void Simulator::setStartingPositions()
{
    const size_t numBytesToCopy = 3 * sizeof(double) * bubbleData.getWidth();

    double *src = bubbleData.getRowPtr((size_t)BP::X);
    double *dst = bubbleData.getRowPtr((size_t)BP::X_START);

    CUDA_CALL(cudaMemcpy(dst, src, numBytesToCopy, cudaMemcpyDeviceToDevice));
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
    ss << properties.getSnapshotFilename() << "." << writer->GetDefaultFileExtension() << "." << numSnapshots;
    writer->SetFileName((ss.str()).c_str());

    // Time stamp
    const double scaledTime = simulationTime * properties.getKParameter() / (properties.getAvgRad() * properties.getAvgRad());
    timeArray->SetNumberOfTuples(1);
    timeArray->SetTuple1(0, scaledTime);
    timeArray->SetName("Time");
    dataSet->GetFieldData()->AddArray(timeArray);

    // Points
    const size_t numComp = 7;
    const size_t memoryStride = bubbleData.getWidth();
    double *devX = bubbleData.getRowPtr((size_t)BP::X);
    hostData.clear();
    hostData.resize(memoryStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), devX, sizeof(double) * numComp * memoryStride, cudaMemcpyDeviceToHost));
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
        t[0] = hostData[i + 0 * memoryStride];
        t[1] = hostData[i + 1 * memoryStride];
        t[2] = hostData[i + 2 * memoryStride];
        points->SetPoint(i, t.data());

        radiiArray->InsertValue(i, hostData[i + 3 * memoryStride]);

        t[0] = hostData[i + 4 * memoryStride];
        t[1] = hostData[i + 5 * memoryStride];
        t[2] = hostData[i + 6 * memoryStride];
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
