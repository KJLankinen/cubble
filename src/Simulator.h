// -*- C++ -*-

#pragma once

#include <string>
#include <vector>
#include <random>
#include <memory>

#include "include/json.hpp"

#include "Vec.h"
#include "BubbleManager.h"

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

	// Integration functions
	
	// Setup functions
	void setupBubbles();

	// Auxiliary functions
	double getSimulationBoxVolume();
	size_t getCellIndexFromNormalizedPosition(const dvec &pos, size_t numCellsPerDim);
	uvec getCellIndexVecFromCellIndex(size_t cellIndex,
					  size_t numCellsPerDim);
	
	size_t getCellIndexFromCellIndexVec(ivec cellIndexVec,
					    int numCellsPerDim);
	void updateNearestNeighbors();
	void removeIntersectingBubbles();
	double getBubbleVolume();
	dvec getScaledPosition(const dvec &position);
	void compressSimulationBox();

	// Parameter & io functions
	void readWriteParameters(bool read);

	// ----------
	// Parameters
	//-----------
	std::string inputFile;
	std::string outputFile;
	std::string saveFile;

	size_t numBubblesPerSweep = 0;
	size_t numMaxSweeps = 0;
	size_t numMaxBubbles = 0;

	int rngSeed = 0;
	
	double avgRad = 0.0;
	double stdDevRad = 0.0;
        double minRad = 0.0;
	double maxRadius = -1.0;
	double phiTarget = 0.0;
	double muZero = 0.0;
	double sigmaZero = 0.0;
	double fZeroPerMuZero = 0.0;
	double errorTolerance = 0.0;
	double timeStep = 0.0;
	double compressionAmount = 0.0;

	// Two vectors that define the volume of the simulation area:
	// lbb = left bottom back corner
	// tfr = top front right corner
	dvec lbb;
	dvec tfr;

	std::shared_ptr<BubbleManager> bubbleManager;

	std::vector<std::vector<size_t>> nearestNeighbors;
    };
}
