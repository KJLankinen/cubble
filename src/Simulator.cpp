#include <iostream>

#include "Simulator.h"
#include "Fileio.h"

using namespace cubble;

Simulator::Simulator(const std::string &inF,
		     const std::string &outF,
		     const std::string &saveF)
{
    env = std::make_shared<Env>(inF, outF, saveF);
    env->readParameters();
    
    bubbleManager = std::make_shared<BubbleManager>();
    cudaKernelWrapper = std::make_shared<CudaKernelWrapper>(bubbleManager, env);
}

Simulator::~Simulator()
{
    env->writeParameters();
}

void Simulator::run()
{   
    cudaKernelWrapper->generateBubblesOnGPU();

    std::string outputFile;
    env->getOutputFile(outputFile);
    
    std::vector<Bubble> b;
    bubbleManager->getBubbles(b);
    
    fileio::writeVectorToFile(outputFile, b);
}
