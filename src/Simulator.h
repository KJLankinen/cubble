// -*- C++ -*-

#pragma once

#include <memory>

#include "BubbleManager.h"
#include "CudaKernelWrapper.h"
#include "Env.h"

namespace cubble
{
    class Simulator
    {
    public:
        Simulator(const std::string &inputFile,
		  const std::string &outputFile,
		  const std::string &saveFile);
	~Simulator();
	void run();
	
    private:
	std::shared_ptr<BubbleManager> bubbleManager;
	std::shared_ptr<CudaKernelWrapper> cudaKernelWrapper;
	std::shared_ptr<Env> env;
    };
}
