#include <iostream>

#include "Integrator.h"
#include "Fileio.h"
#include "Macros.h"

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
    
    readWriteParameters(true);

    bubbles.reserve(numBubbles);    
    normDist = ndd(avgRad, stdDevRad);
}

Integrator::~Integrator()
{
    readWriteParameters(false);
}

void Integrator::run()
{
    // Start the simulation.
    // Do the necessary pre-simulation steps and then start
    // the 'main loop'.

    for (size_t i = 0; i < numBubbles; ++i)
	generateBubble(Vector3<double>(0, 0, 0), Vector3<double>(1, 1, 1));

    fileio::writeVectorToFile(outputFile, bubbles);
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
    double radius = normDist(generator);
    while (radius < minRad)
	radius = normDist(generator);
    
    bubbles.emplace_back(position, radius);
}

void Integrator::integrate(double dt)
{
    phi = dt * 0.1;
}

void Integrator::readWriteParameters(bool read)
{
    std::string msg = read
	? "Reading parameters from file " + inputFile
	: "Saving parameters to file " + saveFile;
    
    std::cout << msg << std::endl;

    nlohmann::json params;

    if (read)
	fileio::readFileToJSON(inputFile, params);
    
    _PARAMETERIZE(read, phi, params);
    _PARAMETERIZE(read, avgRad, params);
    _PARAMETERIZE(read, stdDevRad, params);
    _PARAMETERIZE(read, minRad, params);
    _PARAMETERIZE(read, numBubbles, params);

    if (!read)
	fileio::writeJSONToFile(saveFile, params);
}
