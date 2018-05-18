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

	// Setup functions
	void setupSimulation();
	
	// Integration functions
	void integrate();

	// Auxiliary functions
	dvec wrapAroundBoundaries(dvec position);
	double getSimulationBoxVolume();
	size_t getCellIndexFromNormalizedPosition(const dvec &pos);
	uvec getCellIndexVecFromCellIndex(size_t cellIndex);
	size_t getCellIndexFromCellIndexVec(ivec cellIndexVec);
	void removeIntersectingBubbles();
	double getBubbleVolume();
	dvec getScaledPosition(const dvec &position);
	dvec getNormalizedPosition(const dvec &position);
	void compressSimulationBox();
	void addNeighborCellsToVec(std::vector<size_t> &v,
				   size_t cellIndex);
	void resetCellData();
	void updateCellDataForBubble(size_t i, dvec position);
	
	// Parameter & io functions
	void readWriteParameters(bool read);

	// ----------
	// Parameters
	//-----------
	std::string inputFile;
	std::string outputFile;
	std::string saveFile;

	// Parameters read from input file
	size_t numBubblesPerSweep = 0;
	size_t numMaxSweeps = 0;
	size_t numMaxBubbles = 0;
	size_t numIntegrationSteps = 0;

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

	// Internal variables.
	size_t numCellsPerDim = 1;
	double maxDiameter = -1.0;

	// Two vectors that define the volume of the simulation area:
	// lbb = left bottom back corner
	// tfr = top front right corner
	dvec lbb;
	dvec tfr;

	std::shared_ptr<BubbleManager> bubbleManager;

	std::vector<dvec> accelerations;
	std::vector<std::vector<size_t>> cellToBubbles;
	std::vector<std::vector<size_t>> bubbleToCells;
    };
}
