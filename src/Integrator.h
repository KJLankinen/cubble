// -*- C++ -*-

#pragma once

#include <string>
#include <vector>
#include <random>

#include "include/json.hpp"

typedef std::uniform_real_distribution<double> urdd;
typedef std::normal_distribution<double> ndd;

namespace cubble
{
    // This class is the workhorse of the simulation.
    class Integrator
    {
    public:
	Integrator(const std::string &inputFile,
		   const std::string &outputFile,
		   const std::string &saveFile,
		   const int rngSeed = 0);
	~Integrator();
	void run();
	
    private:
	//----------
	// Functions
	//----------
	
	// Setup functions
	void generateBubble();

	// Auxiliary functions
	double getSimulationBoxVolume();
	size_t getCellIndexFromPos(const Vector3<double> &pos, size_t numCellsPerDim);
	Vector3<size_t> getCellIndexVecFromCellIndex(size_t cellIndex,
						     size_t numCellsPerDim);
	size_t getCellIndexFromCellIndexVec(Vector3<size_t> cellIndexVec,
					    size_t numCellsPerDim);
	void updateNearestNeighbors();
	void removeIntersectingBubbles();

	// Integration functions

	// Parameter & io functions
	void readWriteParameters(bool read);

	// ----------
	// Parameters
	//-----------
	urdd uniDist;
	ndd normDist;
	std::mt19937 generator;
	
	std::string inputFile;
	std::string outputFile;
	std::string saveFile;

	size_t numBubbles = 0;
	size_t dataStride = 4;

	double maxRadius = -1;
	double phiTarget = 0.0;
	double muZero = 0.0;
	double sigmaZero = 0.0;
	double fZeroPerMuZero = 0.0;
	double avgRad = 0.0;
	double stdDevRad = 0.0;
	double minRad = 0.0;
	double errorTolerance = 0.0;
	double timeStep = 0.0;

	// Two vectors that define the volume of the simulation area:
	// lbb = left bottom back corner
	// tfr = top front right corner
	Vector3<double> lbb;
	Vector3<double> tfr;

	std::vector<double> bubbleData;
	std::vector<std::vector<size_t>> nearestNeighbors;
	std::vector<std::vector<size_t>> tentativeNearestNeighbors;
    };
}
