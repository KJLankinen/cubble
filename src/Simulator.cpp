#include <iostream>
#include <sstream>
#include <set>

#include "Simulator.h"
#include "Fileio.h"
#include "Macros.h"
#include "CudaContainer.h"

using namespace cubble;

Simulator::Simulator(const std::string &inF,
		     const std::string &outF,
		     const std::string &saveF)
{
    inputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + inF);
    outputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + saveF);
    
    readWriteParameters(true);

    dvec interval = tfr - lbb;
    assert(interval.x == interval.y && "Simulation box must be a square or a cube!");
#if(NUM_DIM == 3)
    assert(interval.x == interval.z && "Simulation box must be a square or a cube!");
#endif
    
    fZeroPerMuZero = sigmaZero * avgRad / muZero;
    bubbleManager = std::make_shared<BubbleManager>();
    cudaKernelWrapper = std::make_shared<CudaKernelWrapper>(bubbleManager);
}

Simulator::~Simulator()
{
    readWriteParameters(false);
}

void Simulator::run()
{
    cudaKernelWrapper->generateBubblesOnGPU(numBubbles,
					    numCells,
					    rngSeed,
					    avgRad,
					    stdDevRad,
					    lbb,
					    tfr);
    
    fileio::writeVectorToFile(outputFile, bubbleManager->bubbles);
}

void Simulator::readWriteParameters(bool read)
{
    std::string msg = read
	? "Reading parameters from file " + inputFile
	: "Saving parameters to file " + saveFile;
    
    std::cout << msg << "\n" << std::endl;

    nlohmann::json params;

    if (read)
	fileio::readFileToJSON(inputFile, params);

    // When adding new parameters, be sure to add them to the input .json as well
    // and with the exact same name as here.
    
    CUBBLE_PARAMETER(read, params, phiTarget);
    CUBBLE_PARAMETER(read, params, muZero);
    CUBBLE_PARAMETER(read, params, sigmaZero);
    CUBBLE_PARAMETER(read, params, avgRad);
    CUBBLE_PARAMETER(read, params, stdDevRad);
    CUBBLE_PARAMETER(read, params, minRad);
    CUBBLE_PARAMETER(read, params, lbb);
    CUBBLE_PARAMETER(read, params, tfr);
    CUBBLE_PARAMETER(read, params, errorTolerance);
    CUBBLE_PARAMETER(read, params, timeStep);
    CUBBLE_PARAMETER(read, params, rngSeed);
    CUBBLE_PARAMETER(read, params, compressionAmount);
    CUBBLE_PARAMETER(read, params, numIntegrationSteps);
    CUBBLE_PARAMETER(read, params, numBubbles);
    CUBBLE_PARAMETER(read, params, numCells);
    
    if (!read)
	fileio::writeJSONToFile(saveFile, params);

    std::cout << "\nParameter IO done." << std::endl;
}
