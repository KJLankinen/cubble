// -*- C++ -*-

#pragma once

#include "Bubble.h"

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
	void generateBubble(Vector3<double> intervalStart, Vector3<double> intervalEnd);
	void integrate(double dt);
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

	std::vector<Bubble> bubbles;

	size_t numBubbles = 0;
	
	double phi = 0.0;
	double avgRad = 0.0;
	double stdDevRad = 0.0;
	double minRad = 0.0;
    };
}
