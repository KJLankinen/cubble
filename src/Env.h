// -*- C++ -*-

#pragma once

#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "json.hpp"

#include "Macros.h"
#include "Vec.h"
#include "Globals.h"

namespace cubble
{
class Env
{
	// See Macros.h for details of this macro
	CUBBLE_CONST_PROP(int, NumBubblesPerCell, 0)
	CUBBLE_CONST_PROP(int, RngSeed, 0)
	CUBBLE_CONST_PROP(int, NumStepsToRelax, 0)
	CUBBLE_CONST_PROP(int, NumBubbles, 0)
	CUBBLE_CONST_PROP(int, MinNumBubbles, 0)

	CUBBLE_CONST_PROP(CubbleFloatType, AvgRad, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, StdDevRad, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, MinRad, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, PhiTarget, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, MuZero, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, SigmaZero, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, FZeroPerMuZero, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, ErrorTolerance, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, MaxDeltaEnergy, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, KParameter, 0)
	CUBBLE_CONST_PROP(CubbleFloatType, Kappa, 0)
	CUBBLE_PROP(CubbleFloatType, TimeStep, 0)

	CUBBLE_CONST_PROP(std::string, SnapshotFilename, "")
	CUBBLE_CONST_PROP(std::string, DataFilename, "")

	CUBBLE_PROP(fpvec, Lbb, fpvec(0, 0, 0))
	CUBBLE_PROP(fpvec, Tfr, fpvec(0, 0, 0))
	CUBBLE_PROP(fpvec, BoxRelativeDimensions, fpvec(0, 0, 0))

  public:
	Env(const std::string &inF,
		const std::string &saveF)
	{
		inputFile = std::string(inF);
		saveFile = std::string(saveF);
	}

	~Env() {}
	void readParameters()
	{
		readWriteParameters(true);

		// Calculate 'derived' parameters after reading.
		assert(MuZero > 0);
		assert(BoxRelativeDimensions.x > 0);
		assert(BoxRelativeDimensions.y > 0);
		assert(BoxRelativeDimensions.z > 0);
		FZeroPerMuZero = SigmaZero * AvgRad / MuZero;

		MinRad = 0.1 * AvgRad;
	}

	void writeParameters() { readWriteParameters(false); }

	CubbleFloatType getSimulationBoxVolume() const
	{
		fpvec temp = Tfr - Lbb;
#if NUM_DIM == 3
		return temp.x * temp.y * temp.z;
#else
		return temp.x * temp.y;
#endif
	}

	CubbleFloatType getPi() const { return 3.1415926535897932384626433832795028841971693993; }

  private:
	void readWriteParameters(bool read)
	{
		std::string msg = read
							  ? "\nReading parameters from file " + inputFile
							  : "\nSaving parameters to file " + saveFile;

		std::cout << msg << std::endl;
		nlohmann::json params;
		std::fstream file;

		if (read)
		{
			file.open(inputFile, std::ios::in);
			file >> params;
		}

		// When adding new parameters, be sure to add them to the input .json as well
		// and with the exact same name as here.

		CUBBLE_IO_PARAMETER(read, params, PhiTarget);
		CUBBLE_IO_PARAMETER(read, params, MuZero);
		CUBBLE_IO_PARAMETER(read, params, SigmaZero);
		CUBBLE_IO_PARAMETER(read, params, AvgRad);
		CUBBLE_IO_PARAMETER(read, params, StdDevRad);
		CUBBLE_IO_PARAMETER(read, params, ErrorTolerance);
		CUBBLE_IO_PARAMETER(read, params, TimeStep);
		CUBBLE_IO_PARAMETER(read, params, RngSeed);
		CUBBLE_IO_PARAMETER(read, params, NumBubblesPerCell);
		CUBBLE_IO_PARAMETER(read, params, SnapshotFilename);
		CUBBLE_IO_PARAMETER(read, params, NumStepsToRelax);
		CUBBLE_IO_PARAMETER(read, params, MaxDeltaEnergy);
		CUBBLE_IO_PARAMETER(read, params, KParameter);
		CUBBLE_IO_PARAMETER(read, params, NumBubbles);
		CUBBLE_IO_PARAMETER(read, params, Kappa);
		CUBBLE_IO_PARAMETER(read, params, MinNumBubbles);
		CUBBLE_IO_PARAMETER(read, params, DataFilename);
		CUBBLE_IO_PARAMETER(read, params, BoxRelativeDimensions);

		if (!read)
		{
			file.open(saveFile, std::ios::out);
			file << std::setw(4) << params << std::endl;
		}
	}

	std::string inputFile;
	std::string saveFile;

	CubbleFloatType integrationTime = 0.0;
};
}; // namespace cubble
