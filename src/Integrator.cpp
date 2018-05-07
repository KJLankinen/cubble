#include <iostream>

#include "Integrator.h"
#include "Fileio.h"
#include "Util.h"

#include "include/json.hpp"

using namespace cubble;

Integrator::Integrator(const std::string &inF,
		       const std::string &outF,
		       const std::string &saveF,
		       const int rngSeed)
{
    inputFile = std::string(_XSTRINGIFY(DATA_PATH) + inF);
    outputFile = std::string(_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(_XSTRINGIFY(DATA_PATH) + saveF);

    generator = std::mt19937(rngSeed);
    uniDist = urdd(0, 1);
    
    deserialize();

    bubbles.reserve(numBubbles);    
    normDist = ndd(avgRad, stdDevRad);
}

Integrator::~Integrator()
{
    serialize();
}

void Integrator::run()
{
    // Start the simulation.
    // Do the necessary pre-simulation steps and then start
    // the 'main loop'.

    for (size_t i = 0; i < numBubbles; ++i)
	generateBubble(Vector3<double>(0, 0, 0), Vector3<double>(1, 1, 1));

    for (const Bubble &b : bubbles)
	std::cout << b << std::endl;
}

void Integrator::generateBubble(Vector3<double> intervalStart, Vector3<double> intervalEnd)
{
    auto generatePosition = [=]()
	{
	    Vector3<double> position(uniDist(generator),
				     uniDist(generator),
				     uniDist(generator));
	    
	    position *= intervalEnd - intervalStart;
	    position += intervalStart;

	    return position;
	};

    Vector3<double> position = generatePosition();
    double radius = normDist(generator) * avgRad;
    while (radius < minRad)
	radius = normDist(generator) * avgRad;
    
    bubbles.emplace_back(position, radius);
}

void Integrator::integrate(double dt)
{
    phi = dt * 0.1;
}

void Integrator::deserialize()
{
    std::cout << "Reading state from file..." << std::endl;
    
    json inputParams;
    fileio::readFileToJSON(inputFile, inputParams);

    phi = inputParams["phi"];
    avgRad = inputParams["avgRad"];
    stdDevRad = inputParams["stdDevRad"];
    minRad = inputParams["minRad"];
    numBubbles = inputParams["numBubbles"];
}

void Integrator::serialize()
{
    std::cout << "Writing state to file..." << std::endl;
    
    json outputParams;
    outputParams["phi"] = phi;
    outputParams["avgRad"] = avgRad;
    outputParams["stdDevRad"] = stdDevRad;
    outputParams["minRad"] = minRad;
    outputParams["numBubbles"] = numBubbles;

    fileio::writeJSONToFile(saveFile, outputParams);
}
