#include <iostream>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include "CubbleApp.h"
#include "Fileio.h"
#include "Bubble.h"

using namespace cubble;

int CubbleApp::numSnapshots = 0;

CubbleApp::CubbleApp(const std::string &inF, const std::string &saveF)
{
    env = std::make_shared<Env>(inF, saveF);
    env->readParameters();

    simulator = std::make_unique<Simulator>(env);
}

CubbleApp::~CubbleApp()
{
    CUDA_CALL(cudaDeviceSynchronize());
}

void CubbleApp::run()
{
    try
    {
        setupSimulation();
        stabilizeSimulation();
        runSimulation();
    }
    catch (const std::runtime_error &err)
    {
        std::cout << "Runtime error encountered! Saving a final snapshot and parameters." << std::endl;
        saveSnapshotToFile();
        env->writeParameters();

        throw err;
    }

    saveSnapshotToFile();
    env->writeParameters();

    std::cout << "Simulation has been finished.\nGoodbye!" << std::endl;
}

void CubbleApp::setupSimulation()
{
    std::cout << "======\nSetup\n======" << std::endl;

    simulator->setupSimulation();
    saveSnapshotToFile();

    std::cout << "Letting bubbles settle after they've been created and before scaling or stabilization." << std::endl;
    for (size_t i = 0; i < (size_t)env->getNumStepsToRelax(); ++i)
        simulator->integrate();

    saveSnapshotToFile();

    const double phiTarget = env->getPhiTarget();
    double bubbleVolume = simulator->getVolumeOfBubbles();
    double phi = bubbleVolume / env->getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;

    simulator->transformPositions(true);
    const dvec relativeSize = env->getBoxRelativeDimensions();
#if (NUM_DIM == 3)
    const double t = std::cbrt(simulator->getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z));
#else
    const double t = std::sqrt(simulator->getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y));
#endif
    env->setTfr(dvec(t, t, t) * relativeSize);
    simulator->transformPositions(false);

    phi = bubbleVolume / env->getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi
              << ", target: " << phiTarget
              << std::endl;

    saveSnapshotToFile();
}

void CubbleApp::stabilizeSimulation()
{
    std::cout << "=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    simulator->integrate();
    simulator->calculateEnergy();
    double energy2 = simulator->getElasticEnergy();

    while (true)
    {
        double energy1 = energy2;
        double time = 0;

        for (int i = 0; i < env->getNumStepsToRelax(); ++i)
        {
            simulator->integrate();
            time += env->getTimeStep();
        }

        simulator->calculateEnergy();
        energy2 = simulator->getElasticEnergy();
        double deltaEnergy = std::abs(energy2 - energy1) / time;
        deltaEnergy *= 0.5 * env->getSigmaZero();

        if (deltaEnergy < env->getMaxDeltaEnergy())
        {
            std::cout << "Final delta energy " << deltaEnergy
                      << " after " << (numSteps + 1) * env->getNumStepsToRelax()
                      << " steps."
                      << " Energy before: " << energy1
                      << ", energy after: " << energy2
                      << ", time: " << time * env->getKParameter() / (env->getAvgRad() * env->getAvgRad())
                      << std::endl;
            break;
        }
        else if (numSteps > failsafe)
        {
            std::cout << "Over " << failsafe * env->getNumStepsToRelax()
                      << " steps taken and required delta energy not reached."
                      << " Check parameters."
                      << std::endl;
            break;
        }
        else
            std::cout << "Number of simulation steps relaxed: "
                      << (numSteps + 1) * env->getNumStepsToRelax()
                      << ", delta energy: " << deltaEnergy
                      << ", energy before: " << energy1
                      << ", energy after: " << energy2
                      << std::endl;

        ++numSteps;
    }

    saveSnapshotToFile();
}

void CubbleApp::runSimulation()
{
    std::cout << "==========\nSimulation\n==========" << std::endl;

    simulator->setSimulationTime(0);

    int numSteps = 0;
    int timesPrinted = 0;
    bool stopSimulation = false;

    std::stringstream dataStream;
    dataStream << env->getDataFilename();

    std::string filename(dataStream.str());
    dataStream.clear();
    dataStream.str("");

    while (!stopSimulation)
    {
        if (numSteps == 2000)
        {
            CUDA_PROFILER_START();
        }

        stopSimulation = !simulator->integrate(true);

        if (numSteps == 2050)
        {
            CUDA_PROFILER_STOP();
#if (USE_PROFILING == 1)
            break;
#endif
        }

        double scaledTime = simulator->getSimulationTime() * env->getKParameter() / (env->getAvgRad() * env->getAvgRad());
        if ((int)scaledTime >= timesPrinted)
        {
            double phi = simulator->getVolumeOfBubbles() / env->getSimulationBoxVolume();
            double relativeRadius = simulator->getAverageRadius() / env->getAvgRad();
            dataStream << scaledTime
                       << " " << relativeRadius
                       << " " << simulator->getMaxBubbleRadius() / env->getAvgRad()
                       << " " << simulator->getNumBubbles()
                       << " " << 1.0 / (simulator->getInvRho() * env->getAvgRad())
                       << "\n";

            std::cout << "t*: " << scaledTime
                      << " <R>/<R_in>: " << relativeRadius
                      << " #b: " << simulator->getNumBubbles()
                      << " phi: " << phi
                      << std::endl;

            // Only write snapshots when t* is a power of 2.
            if ((timesPrinted & (timesPrinted - 1)) == 0)
                saveSnapshotToFile();

            ++timesPrinted;
        }

        ++numSteps;
    }

    fileio::writeStringToFile(filename, dataStream.str());
}

void CubbleApp::saveSnapshotToFile()
{
#if (USE_PROFILING == 1)
    return;
#endif

    std::cout << "Writing a snapshot to a file." << std::endl;
    // This could easily be parallellized s.t. bubbles are fetched serially, but written to file parallelly.

    std::vector<Bubble> tempVec;
    simulator->getBubbles(tempVec);

    std::stringstream ss;
    ss << env->getSnapshotFilename()
       << numSnapshots
       << ".csv";

    std::string filename(ss.str());
    ss.clear();
    ss.str("");

    ss << "x, y, z, r";

    for (const auto &bubble : tempVec)
        ss << "\n"
           << bubble;

    fileio::writeStringToFile(filename, ss.str());
    ++numSnapshots;
}
