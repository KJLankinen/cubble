// -*- C++ -*-

#pragma once

#include <string>
#include <vector>
#include <random>
#include <memory>

#ifndef __CUDACC__
  #include "include/json.hpp"
#endif

#include "Vec.h"
#include "BubbleManager.h"
#include "CudaKernelWrapper.h"

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
	//----------
	// Functions
	//----------
	
	// Parameter & io functions
	void readWriteParameters(bool read);

	// ----------
	// Parameters
	//-----------
	std::string inputFile;
	std::string outputFile;
	std::string saveFile;

	// Parameters read from input file
	size_t numIntegrationSteps = 0;
	size_t numBubbles = 0;
	size_t numCells = 0;

	int rngSeed = 0;
	
	double avgRad = 0.0;
	double stdDevRad = 0.0;
        double minRad = 0.0;
	double phiTarget = 0.0;
	double muZero = 0.0;
	double sigmaZero = 0.0;
	double fZeroPerMuZero = 0.0;
	double errorTolerance = 0.0;
	double timeStep = 0.0;
	double compressionAmount = 0.0;
	double integrationTime = 0.0;

	// Two vectors that define the volume of the simulation area:
	// lbb = left bottom back corner
	// tfr = top front right corner
	dvec lbb;
	dvec tfr;

	std::shared_ptr<BubbleManager> bubbleManager;
	std::shared_ptr<CudaKernelWrapper> cudaKernelWrapper;
    };
}
