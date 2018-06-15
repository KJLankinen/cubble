#include <iostream>

#include "CubbleApp.h"
#include "Fileio.h"
#include "Bubble.h"

using namespace cubble;

int CubbleApp::numSnapshots = 0;

CubbleApp::CubbleApp(const std::string &inF,
		     const std::string &saveF)
{
    env = std::make_shared<Env>(inF, saveF);
    env->readParameters();
    
    simulator = std::make_unique<Simulator>(env);
}

CubbleApp::~CubbleApp()
{
    env->writeParameters();
}

void CubbleApp::run()
{
    simulator->setupSimulation();

    double volRatio = simulator->getVolumeOfBubbles() / (env->getSimulationBoxVolume());
    std::cout << "Current volume ratio: " << volRatio
	      << ", target volume ratio: " << env->getPhiTarget()
	      << std::endl;
    
    saveSnapshotToFile();
}

void CubbleApp::saveSnapshotToFile()
{
    std::cout << "Writing a snap shot to a file..." << std::flush;

    std::vector<Bubble> temp;
    simulator->getBubbles(temp);
    
    std::stringstream ss;
    ss << env->getDataPath()
       << env->getSnapshotFilename()
       << numSnapshots
       << ".dat";
    
    fileio::writeVectorToFile(ss.str(), temp);
    ++numSnapshots;

    std::cout << " Done." << std::endl;
}
