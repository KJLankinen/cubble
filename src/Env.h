// -*- C++ -*-

#pragma once

#include <string>
#include <assert.h>
#include <iostream>

#ifndef __CUDACC__
  #include "include/json.hpp"
#endif

#include "Macros.h"
#include "Fileio.h"
#include "Vec.h"

namespace cubble
{
    class Env
    {
	// See Macros.h for details of this macro
	CUBBLE_PROP(int, NumIntegrationSteps)
	CUBBLE_PROP(int, NumBubbles)
	CUBBLE_PROP(int, NumCellsPerDim)
	CUBBLE_PROP(int, RngSeed)
	
	CUBBLE_PROP(double, AvgRad)
	CUBBLE_PROP(double, StdDevRad)
	CUBBLE_PROP(double, MinRad)
	CUBBLE_PROP(double, PhiTarget)
	CUBBLE_PROP(double, MuZero)
	CUBBLE_PROP(double, SigmaZero)
	CUBBLE_PROP(double, FZeroPerMuZero)
	CUBBLE_PROP(double, ErrorTolerance)
	CUBBLE_PROP(double, TimeStep)
	CUBBLE_PROP(double, CompressionAmount)

	CUBBLE_PROP(dvec, Lbb)
	CUBBLE_PROP(dvec, Tfr)
	
    public:
	Env(const std::string &inF,
	    const std::string &outF,
	    const std::string &saveF)
	{
	    inputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + inF);
	    outputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + outF);
	    saveFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + saveF);
	}
	
	~Env() {}
	
#ifndef __CUDACC__   	
	void readParameters()
	{
	    readWriteParameters(true);
	    
	    // Calculate 'derived' parameters after reading.
	    FZeroPerMuZero = SigmaZero * AvgRad / MuZero;

	    // Perform any parameter related sanity & correctness checking here.
	    dvec interval = Tfr - Lbb;
	    assert(interval.x == interval.y
		   && "Simulation box must be a square or a cube!");
#if(NUM_DIM == 3)
	    assert(interval.x == interval.z
		   && "Simulation box must be a square or a cube!");
#endif
	}
	
	void writeParameters() {readWriteParameters(false); }
#endif
	
	void getOutputFile(std::string &f) { f = outputFile; }
	
    private:
	
#ifndef __CUDACC__   
	void readWriteParameters(bool read)
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
	    
	    CUBBLE_IO_PARAMETER(read, params, PhiTarget);
	    CUBBLE_IO_PARAMETER(read, params, MuZero);
	    CUBBLE_IO_PARAMETER(read, params, SigmaZero);
	    CUBBLE_IO_PARAMETER(read, params, AvgRad);
	    CUBBLE_IO_PARAMETER(read, params, StdDevRad);
	    CUBBLE_IO_PARAMETER(read, params, MinRad);
	    CUBBLE_IO_PARAMETER(read, params, Lbb);
	    CUBBLE_IO_PARAMETER(read, params, Tfr);
	    CUBBLE_IO_PARAMETER(read, params, ErrorTolerance);
	    CUBBLE_IO_PARAMETER(read, params, TimeStep);
	    CUBBLE_IO_PARAMETER(read, params, RngSeed);
	    CUBBLE_IO_PARAMETER(read, params, CompressionAmount);
	    CUBBLE_IO_PARAMETER(read, params, NumIntegrationSteps);
	    CUBBLE_IO_PARAMETER(read, params, NumBubbles);
	    CUBBLE_IO_PARAMETER(read, params, NumCellsPerDim);
	    
	    if (!read)
		fileio::writeJSONToFile(saveFile, params);
	    
	    std::cout << "\nParameter IO done." << std::endl;
	}
#endif
	
	std::string outputFile;
	std::string inputFile;
	std::string saveFile;
	
	double integrationTime = 0.0;
    };
};
