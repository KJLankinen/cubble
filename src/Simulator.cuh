// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Vec.h"
#include "DeviceArray.h"
#include "PinnedHostArray.h"
#include "CubWrapper.h"
#include "Util.h"

#include <utility>
#include <memory>
#include <vector>

namespace cubble
{
enum class BubbleProperty;

class Simulator
{
  public:
	bool init(const char *inputFileName, const char *outputFileName);
	void deinit();
	void run();

  private:
	void setupSimulation();
	bool integrate(bool useGasExchange = false);
	void calculateEnergy();
	double getVolumeOfBubbles();
	double getMaxBubbleRadius() const { return maxBubbleRadius; }
	double getInvRho();
	void transformPositions(bool normalize);
	double getAverageProperty(BubbleProperty property);
	void setStartingPositions();
	void doPrediction(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaEvent_t &eventToMark);
	void doCorrection(const ExecutionPolicy &policy, double timeStep, bool useGasExchange, cudaStream_t &streamThatShouldWait);
	void doGasExchange(ExecutionPolicy policy, const cudaEvent_t &eventToWaitOn, cudaStream_t &streamThatShouldWait);
	void doVelocity(const ExecutionPolicy &policy);
	void doReset(const ExecutionPolicy &policy);
	double doError();
	void doBoundaryWrap(const ExecutionPolicy &policy);
	void doBubbleSizeChecks(ExecutionPolicy policy, cudaStream_t &streamToUse, cudaEvent_t &eventToMark);

	void generateBubbles();
	void updateCellsAndNeighbors();
	void updateData();
	void deleteSmallBubbles(int numBubblesAboveMinRad);
	dim3 getGridSize();
	void saveSnapshotToFile();

	double simulationTime = 0.0;
	double elasticEnergy = 0.0;
	uint32_t numSnapshots = 0;
	int numBubbles = 0;
	ivec bubblesPerDimAtStart = ivec(0, 0, 0);
	size_t integrationStep = 0;

	double maxBubbleRadius = 0;

	cudaEvent_t blockingEvent1;
	cudaEvent_t blockingEvent2;
	cudaStream_t nonBlockingStream1;
	cudaStream_t nonBlockingStream2;

	std::vector<cudaStream_t> neighborStreamVec;
	std::vector<cudaEvent_t> neighborEventVec;

	// Host pointers to device global variables
	int *mbpc = nullptr;
	int *np = nullptr;
	double *dtfapr = nullptr;
	double *dtfa = nullptr;
	double *dvm = nullptr;
	double *dtv = nullptr;
	double *dir = nullptr;
	double *dta = nullptr;
	double *dasai = nullptr;

	Env properties = {};
	std::shared_ptr<CubWrapper> cubWrapper;

	DeviceArray<double> bubbleData;
	DeviceArray<int> aboveMinRadFlags;
	DeviceArray<int> cellData;
	DeviceArray<int> bubbleCellIndices;
	DeviceArray<int> pairs;
	DeviceArray<bool> wrappedFlags;

	PinnedHostArray<int> pinnedInt;
	PinnedHostArray<double> pinnedDouble;

	std::vector<double> hostData;
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

	X_START,
	Y_START,
	Z_START,
	PATH_LENGTH,
	SQUARED_DISTANCE,

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

	TEMP1,
	TEMP2,
	TEMP3,
	TEMP4,
	TEMP5,

	NUM_VALUES
};

enum class CellProperty
{
	OFFSET,
	SIZE,

	NUM_VALUES
};
}; // namespace cubble
