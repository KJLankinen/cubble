// -*- C++ -*-

#pragma once

#include "Bubble.h"
#include "Cell.h"

#include <string>
#include <vector>
#include <map>
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
	void prepareCells();
	void generateBubble();
	void removeBubble(const Bubble &bubble);
	void removeIntersectingBubbles();
	void applyBoundaryConditionsForBubble(Bubble &b);

	// Auxiliary functions
	double getSimulationBoxVolume();
	size_t getCellIndexForPosition(Vector3<double> pos);

	// Integration functions
	void integrate(double dt);
	void computeForces();

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

	std::vector<Cell> cells;
	std::map<size_t, Bubble> bubbles;

	size_t numBubbles = 0;
	size_t cellsPerDim = 1;
	
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
    };
}
