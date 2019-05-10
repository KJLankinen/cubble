// -*- C++ -*-

#pragma once

#include <string>
#include <assert.h>
#include "json.hpp"

#include "Vec.h"

namespace cubble
{
struct SimulationProperties
{
	// From json
	double phiTarget = 0.0;
	double kappa = 0.0;
	int numBubbles = 0;
	int minNumBubbles = 0;
	dvec boxRelativeDimensions = dvec(0, 0, 0);
	double muZero = 0.0;
	double sigmaZero = 0.0;
	double avgRad = 0.0;
	double stdDevRad = 0.0;
	double errorTolerance = 0.0;
	double timeStep = 0.0;
	int rngSeed = 0;
	int numBubblesPerCell = 0;
	int numStepsToRelax = 0;
	double maxDeltaEnergy = 0.0;
	double kParameter = 0.0;
	std::string snapshotFilename = "";
	std::string dataFilename = "";

	// Derived
	double fZeroPerMuZero = 0.0;
	double minRad = 0.0;
};

void to_json(nlohmann::json &j, const SimulationProperties &props)
{
	j = nlohmann::json{
		{"phiTarget", props.phiTarget},
		{"kappa", props.kappa},
		{"numBubbles", props.numBubbles},
		{"minNumBubbles", props.minNumBubbles},
		{"boxRelativeDimensions", props.boxRelativeDimensions},
		{"muZero", props.muZero},
		{"sigmaZero", props.sigmaZero},
		{"avgRad", props.avgRad},
		{"stdDevRad", props.stdDevRad},
		{"errorTolerance", props.errorTolerance},
		{"timeStep", props.timeStep},
		{"rngSeed", props.rngSeed},
		{"numBubblesPerCell", props.numBubblesPerCell},
		{"numStepsToRelax", props.numStepsToRelax},
		{"maxDeltaEnergy", props.maxDeltaEnergy},
		{"kParameter", props.kParameter},
		{"snapshotFilename", props.snapshotFilename},
		{"dataFilename", props.dataFilename},
	};
}

#define PROPERTY_FROM_JSON(j, p, param) j.at(#param).get_to(p.param)
void from_json(const nlohmann::json &j, SimulationProperties &props)
{
	PROPERTY_FROM_JSON(j, props, phiTarget);
	PROPERTY_FROM_JSON(j, props, kappa);
	PROPERTY_FROM_JSON(j, props, numBubbles);
	PROPERTY_FROM_JSON(j, props, minNumBubbles);
	PROPERTY_FROM_JSON(j, props, boxRelativeDimensions);
	PROPERTY_FROM_JSON(j, props, muZero);
	PROPERTY_FROM_JSON(j, props, sigmaZero);
	PROPERTY_FROM_JSON(j, props, avgRad);
	PROPERTY_FROM_JSON(j, props, stdDevRad);
	PROPERTY_FROM_JSON(j, props, errorTolerance);
	PROPERTY_FROM_JSON(j, props, timeStep);
	PROPERTY_FROM_JSON(j, props, rngSeed);
	PROPERTY_FROM_JSON(j, props, numBubblesPerCell);
	PROPERTY_FROM_JSON(j, props, numStepsToRelax);
	PROPERTY_FROM_JSON(j, props, maxDeltaEnergy);
	PROPERTY_FROM_JSON(j, props, kParameter);
	PROPERTY_FROM_JSON(j, props, snapshotFilename);
	PROPERTY_FROM_JSON(j, props, dataFilename);

	assert(props.muZero > 0);
	assert(props.boxRelativeDimensions.x > 0);
	assert(props.boxRelativeDimensions.y > 0);
	assert(props.boxRelativeDimensions.z > 0);

	props.fZeroPerMuZero = props.sigmaZero * props.avgRad / props.muZero;
	props.minRad = 0.1 * props.avgRad;
}
#undef PROPERTY_FROM_JSON
}; // namespace cubble
