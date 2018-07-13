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
	CUBBLE_CONST_PROP(int, NumIntegrationSteps, 0)
	CUBBLE_CONST_PROP(int, NumBubblesPerCell, 0)
	CUBBLE_CONST_PROP(int, RngSeed, 0)
	CUBBLE_CONST_PROP(int, NumStepsToRelax, 0)
	
	CUBBLE_CONST_PROP(double, AvgRad, 0)
	CUBBLE_CONST_PROP(double, StdDevRad, 0)
	CUBBLE_CONST_PROP(double, MinRad, 0)
	CUBBLE_CONST_PROP(double, PhiTarget, 0)
	CUBBLE_CONST_PROP(double, MuZero, 0)
	CUBBLE_CONST_PROP(double, SigmaZero, 0)
	CUBBLE_CONST_PROP(double, FZeroPerMuZero, 0)
	CUBBLE_CONST_PROP(double, ErrorTolerance, 0)
	CUBBLE_CONST_PROP(double, ScaleAmount, 0)
	CUBBLE_CONST_PROP(double, MaxDeltaEnergy, 0)
	CUBBLE_CONST_PROP(double, KParameter, 0)
	CUBBLE_PROP(double, TimeStep, 0)
	
	CUBBLE_CONST_PROP(std::string, DataPath, "")
	CUBBLE_CONST_PROP(std::string, SnapshotFilename, "")

	CUBBLE_PROP(dvec, Lbb, dvec(0, 0, 0))
	CUBBLE_PROP(dvec, Tfr, dvec(0, 0, 0))
	
    public:
	Env(const std::string &inF,
	    const std::string &saveF)
	{
	    DataPath = CUBBLE_XSTRINGIFY(DATA_PATH);
	    inputFile = std::string(DataPath + inF);
	    saveFile = std::string(DataPath + saveF);
	}
	
	~Env() {}
	
#ifndef __CUDACC__   	
	void readParameters()
	{
	    readWriteParameters(true);
	    
	    // Calculate 'derived' parameters after reading.
	    assert(MuZero > 0);
	    FZeroPerMuZero = SigmaZero * AvgRad / MuZero;

	    MinRad = 0.1 * AvgRad;
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

	double getSimulationBoxVolume() const
	{
	    dvec temp = Tfr - Lbb;
#if NUM_DIM == 3
	    return temp.x * temp.y * temp.z;
#else
	    return temp.x * temp.y;
#endif
	}

	double getPi() const { return 3.1415926535897932384626433832795028841971693993; }
	
    private:
	
#ifndef __CUDACC__   
	void readWriteParameters(bool read)
	{
	    std::string msg = read
		? "\nReading parameters from file " + inputFile
		: "\nSaving parameters to file " + saveFile;
	    
	    std::cout << msg << std::endl;
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
	    CUBBLE_IO_PARAMETER(read, params, Lbb);
	    CUBBLE_IO_PARAMETER(read, params, Tfr);
	    CUBBLE_IO_PARAMETER(read, params, ErrorTolerance);
	    CUBBLE_IO_PARAMETER(read, params, TimeStep);
	    CUBBLE_IO_PARAMETER(read, params, RngSeed);
	    CUBBLE_IO_PARAMETER(read, params, ScaleAmount);
	    CUBBLE_IO_PARAMETER(read, params, NumIntegrationSteps);
	    CUBBLE_IO_PARAMETER(read, params, NumBubblesPerCell);
	    CUBBLE_IO_PARAMETER(read, params, SnapshotFilename);
	    CUBBLE_IO_PARAMETER(read, params, NumStepsToRelax);
	    CUBBLE_IO_PARAMETER(read, params, MaxDeltaEnergy);
	    CUBBLE_IO_PARAMETER(read, params, KParameter);
	    
	    if (!read)
		fileio::writeJSONToFile(saveFile, params);
	}
#endif
        
	std::string inputFile;
	std::string saveFile;
	
	double integrationTime = 0.0;
    };
};
