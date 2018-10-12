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

void CubbleApp::run()
{
    setupSimulation();
    stabilizeSimulation();
    runSimulation();
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
        simulator->integratePosition(true);

    saveSnapshotToFile();

    int numSteps = 0;
    const double phiTarget = env->getPhiTarget();
    double bubbleVolume = simulator->getVolumeOfBubbles();
    double phi = bubbleVolume / env->getSimulationBoxVolume();

    auto printPhi = [](double phi, double phiTarget) -> void {
        std::cout << "Volume ratios: current: " << phi
                  << ", target: " << phiTarget
                  << std::endl;
    };

    printPhi(phi, phiTarget);

    std::cout << "Starting the scaling of the simulation box." << std::endl;
    const bool shouldShrink = phi < phiTarget;
    const dvec scaleAmount = env->getScaleAmount() * (shouldShrink ? 1 : -1) * env->getTfr();
    while ((shouldShrink && phi < phiTarget) || (!shouldShrink && phi > phiTarget))
    {
        env->setTfr(env->getTfr() - scaleAmount);

        for (size_t i = 0; i < 50; ++i)
            simulator->integratePosition(true);

        bubbleVolume = simulator->getVolumeOfBubbles();
        phi = bubbleVolume / env->getSimulationBoxVolume();

        if (numSteps % 50 == 0)
            printPhi(phi, phiTarget);

        ++numSteps;
    }

    std::cout << "Scaling took total of " << numSteps << " steps." << std::endl;

    printPhi(phi, phiTarget);
    saveSnapshotToFile();
}

void CubbleApp::stabilizeSimulation()
{
    std::cout << "=============\nStabilization\n=============" << std::endl;

    int numSteps = 0;
    const int failsafe = 500;

    simulator->integratePosition(false);
    double energy2 = simulator->getElasticEnergy();

    while (true)
    {
        double energy1 = energy2;
        double time = 0;

        for (int i = 0; i < env->getNumStepsToRelax(); ++i)
        {
            simulator->integratePosition(false);
            time += env->getTimeStep();
        }

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
    dataStream << env->getDataPath() << env->getDataFilename();

    std::string filename(dataStream.str());
    dataStream.clear();
    dataStream.str("");

    while (!stopSimulation)
    {
        if (numSteps == 2000)
        {
            CUDA_PROFILER_START();
        }

        stopSimulation = !simulator->integrate();

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
            dataStream << scaledTime << " " << relativeRadius << "\n";

            std::cout << "t*: " << scaledTime
                      << " <R>/<R_in>: " << relativeRadius
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
    ss << env->getDataPath()
       << env->getSnapshotFilename()
       << numSnapshots
       << ".dat";

    std::string filename(ss.str());
    ss.clear();
    ss.str("");

    // Add descriptions here, when adding new things to the 'header' of the data file
    ss << "#--------------------------------------------------"
       << "\n# Lines starting with '#' are comment lines"
       << "\n#"
       << "\n# Format of data:"
       << "\n# left bottom back"
       << "\n# top front right"
       << "\n#"
       << "\n# bubble data: (x, y, z, r)"
       << "\n#--------------------------------------------------";

    // Add the new things here.
    ss << "\n"
       << env->getLbb()
       << "\n"
       << env->getTfr();

    for (const auto &bubble : tempVec)
        ss << "\n"
           << bubble;

    fileio::writeStringToFile(filename, ss.str());
    ++numSnapshots;
}
