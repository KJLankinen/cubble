// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "Vec.h"
#include "FixedSizeDeviceArray.h"
#include "CubWrapper.h"

#include <utility>
#include <memory>
#include <vector>

namespace cubble
{
enum class BubbleProperty
{
	X,
	Y,
	Z,
	R,

	X_PRD,
	Y_PRD,
	Z_PRD,
	R_PRD,

	DXDT,
	DYDT,
	DZDT,
	DRDT,

	DXDT_PRD,
	DYDT_PRD,
	DZDT_PRD,
	DRDT_PRD,

	DXDT_OLD,
	DYDT_OLD,
	DZDT_OLD,
	DRDT_OLD,

	ENERGY,
	FREE_AREA,
	ERROR,
	VOLUME,

	NUM_VALUES
};

enum class CellProperty
{
	OFFSET,
	SIZE,

	NUM_VALUES
};

class Simulator
{
	CUBBLE_PROP(double, SimulationTime, 0)
	CUBBLE_PROP(double, ElasticEnergy, 0)
  public:
	Simulator(std::shared_ptr<Env> e);
	~Simulator();

	void setupSimulation();
	bool integrate(bool useGasExchange = false, bool calculateEnergy = false);
	double getVolumeOfBubbles();
	double getAverageRadius();
	void getBubbles(std::vector<Bubble> &bubbles) const;

  private:
	void resetValues();
	void generateBubbles();
	void updateCellsAndNeighbors();
	void updateData();
	bool deleteSmallBubbles();
	dim3 getGridSize();

	int givenNumBubblesPerDim = 0;
	int numBubbles = 0;
	size_t integrationStep = 0;
	int hostNumPairs = 0;

	std::shared_ptr<Env> env;
	std::shared_ptr<CubWrapper> cubWrapper;

	FixedSizeDeviceArray<double> bubbleData;
	FixedSizeDeviceArray<int> aboveMinRadFlags;
	FixedSizeDeviceArray<int> cellData;
	FixedSizeDeviceArray<int> indicesPerCell;
	FixedSizeDeviceArray<int> neighborPairIndices;
	FixedSizeDeviceArray<int> numPairs;

	std::vector<double> hostData;
	std::vector<double *> pointersToArrays;
	std::vector<std::pair<BubbleProperty, BubbleProperty>> pairedProperties;
};
}; // namespace cubble
