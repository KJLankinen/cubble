#include <iostream>

#include "Simulator.h"
#include "Fileio.h"
#include "Bubble.h"

using namespace cubble;

int Simulator::numSnapshots = 0;

Simulator::Simulator(const std::string &inF,
		     const std::string &saveF)
{
    env = std::make_shared<Env>(inF, saveF);
    env->readParameters();
    
    bubbleManager = std::make_shared<BubbleManager>(env);
    cudaKernelWrapper = std::make_shared<CudaKernelWrapper>(bubbleManager, env);
}

Simulator::~Simulator()
{
    env->writeParameters();
}

void Simulator::run()
{
    std::vector<Bubble> temp;
    cudaKernelWrapper->generateBubbles(temp);
    cudaKernelWrapper->assignBubblesToCells(temp);
    cudaKernelWrapper->removeIntersectingBubbles();
    
    bubbleManager->getBubbles(temp);
    cudaKernelWrapper->assignBubblesToCells(temp);

    double volRatio = bubbleManager->getVolumeOfBubbles() / (env->getSimulationBoxVolume());
    std::cout << "Current volume ratio: " << volRatio
	      << ", target volume ratio: " << env->getPhiTarget()
	      << std::endl;
    
    saveSnapshotToFile();
}

void Simulator::saveSnapshotToFile()
{
    std::cout << "Writing a snap shot to a file..." << std::flush;
    std::vector<Bubble> temp;
    bubbleManager->getBubbles(temp);

    std::stringstream ss;
    ss << env->getDataPath()
       << env->getSnapshotFilename()
       << numSnapshots
       << ".dat";
    
    fileio::writeVectorToFile(ss.str(), temp);
    ++numSnapshots;

    std::cout << " Done." << std::endl;
}
