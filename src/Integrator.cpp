#include <iostream>

#include "Integrator.h"
#include "Fileio.h"
#include "Util.h"

#include "include/json.hpp"

using cubble;

Integrator::Integrator(const std::string &inF,
		       const std::string &outF,
		       const std::string &saveF)
{
    inputFile = std::string(_XSTRINGIFY(DATA_PATH) + inF);
    outFile = std::string(_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(_XSTRINGIFY(DATA_PATH) + saveF);
    
    deserialize();
}

Integrator::~Integrator()
{
    serialize();
}

void integrate(double dt)
{
    phi = dt * 0.1;
    alpha = dt + 5.0;
}

void Integrator::deserialize()
{
    std::cout << "Reading state from file..." << std::endl;
    
    json inputParams;
    fileio::readFileToJSON(inputFile, inputParams);

    phi = inputParams["phi"];
    alpha = inputParams["alpha"];
}

void Integrator::serialize()
{
    std::cout << "Writing state to file..." << std::endl;
    
    json outputParams;
    outputParams["phi"] = phi;
    outputParams["alpha"] = alpha;

    fileio::writeJSONToFile(saveFile, outputParams);
}
