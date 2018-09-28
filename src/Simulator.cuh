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
enum class BubbleProperty;

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
	void generateBubbles();
	void updateCellsAndNeighbors();
	void updateData();
	bool deleteSmallBubbles();
	dim3 getGridSize();

	int numBubbles = 0;
	ivec bubblesPerDimAtStart = ivec(0, 0, 0);
	size_t integrationStep = 0;
	int hostNumPairs = 0;

	std::shared_ptr<Env> env;
	std::shared_ptr<CubWrapper> cubWrapper;

	FixedSizeDeviceArray<double> bubbleData;
	FixedSizeDeviceArray<int> aboveMinRadFlags;
	FixedSizeDeviceArray<int> cellData;
	FixedSizeDeviceArray<int> bubbleCellIndices;
	FixedSizeDeviceArray<int> neighborPairIndices;

	std::vector<double> hostData;

	std::vector<std::pair<BubbleProperty, BubbleProperty>> pairedProperties;
};

enum class BubbleProperty
{
	X,
	Y,
	Z,
	R,

	DXDT,
	DYDT,
	DZDT,
	DRDT,

	DXDT_OLD,
	DYDT_OLD,
	DZDT_OLD,
	DRDT_OLD,

	X_PRD,
	Y_PRD,
	Z_PRD,
	R_PRD,

	DXDT_PRD,
	DYDT_PRD,
	DZDT_PRD,
	DRDT_PRD,

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
}; // namespace cubble
