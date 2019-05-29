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
class Simulator
{
  public:
	Simulator(){};
	~Simulator(){};
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
	double getAverageProperty(double *p);

	void generateBubbles();
	void updateCellsAndNeighbors();
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

	Env properties;
	std::shared_ptr<CubWrapper> cubWrapper;

	DeviceArray<int> aboveMinRadFlags;
	DeviceArray<int> cellData;
	DeviceArray<int> bubbleCellIndices;
	DeviceArray<int> pairs;
	DeviceArray<int> wrapMultipliers;

	PinnedHostArray<int> pinnedInt;
	PinnedHostArray<double> pinnedDouble;

	std::vector<double> hostData;

	double *deviceData = nullptr;
	uint32_t dataStride = 0;

	struct AliasedDevicePointers
	{
		// Position & radius
		double *x = nullptr;
		double *y = nullptr;
		double *z = nullptr;
		double *r = nullptr;

		// Change rate (= velocity)
		double *dxdt = nullptr;
		double *dydt = nullptr;
		double *dzdt = nullptr;
		double *drdt = nullptr;

		// Old change rates
		double *dxdtO = nullptr;
		double *dydtO = nullptr;
		double *dzdtO = nullptr;
		double *drdtO = nullptr;

		// Starting coordinates
		double *x0 = nullptr;
		double *y0 = nullptr;
		double *z0 = nullptr;

		// Path length & distance
		double *s = nullptr;
		double *d = nullptr;

		// Predicted coordinates
		double *xP = nullptr;
		double *yP = nullptr;
		double *zP = nullptr;
		double *rP = nullptr;

		// Predicted change rates
		double *dxdtP = nullptr;
		double *dydtP = nullptr;
		double *dzdtP = nullptr;
		double *drdtP = nullptr;

		// Errors of predictions vs corrections
		double *error = nullptr;

		// Dummy for copying stuff & for temporary use
		double *dummy1 = nullptr;
		double *dummy2 = nullptr;
		double *dummy3 = nullptr;
		double *dummy4 = nullptr;
		double *dummy5 = nullptr;
		double *dummy6 = nullptr;
		double *dummy7 = nullptr;
		double *dummy8 = nullptr;
	} adp;
	static const uint32_t numAliases = 34;
	static_assert(sizeof(adp) == sizeof(double *) * numAliases);
};

enum class CellProperty
{
	OFFSET,
	SIZE,

	NUM_VALUES
};
}; // namespace cubble
