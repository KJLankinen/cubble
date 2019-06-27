// -*- C++ -*-

#pragma once

#include "CubWrapper.h"
#include "DeviceArray.h"
#include "Env.h"
#include "PinnedHostArray.h"
#include "Util.h"
#include "Vec.h"
#include <memory>
#include <utility>
#include <vector>

namespace cubble
{
class Simulator
{
public:
  Simulator(){};
  ~Simulator(){};
  void run(const char *inputFileName, const char *outputFileName);

private:
  void setup();
  void deinit();
  double stabilize();
  bool integrate();
  void updateCellsAndNeighbors();
  void deleteSmallBubbles(int numBubblesAboveMinRad);
  void transformPositions(bool normalize);
  double getAverageProperty(double *p);
  dim3 getGridSize();
  void saveSnapshotToFile();
  void reserveMemory();
  void startProfiling(bool start);
  void stopProfiling(bool stop, bool &continueIntegration);
  void doBoundaryWrap(KernelSize ks, int sm, cudaStream_t stream, bool wrapX, bool wrapY,
                      bool wrapZ, int numValues, double *x, double *y, double *z, dvec lbb,
                      dvec tfr, int *mulX, int *mulY, int *mulZ, int *mulOldX, int *mulOldY,
                      int *mulOldZ);
  void doWallVelocity(KernelSize ks, int sm, cudaStream_t stream, bool doX, bool doY, bool doZ,
                      int numValues, int *first, int *second, double *r, double *x, double *y,
                      double *z, double *dxdt, double *dydt, double *dzdt, dvec lbb, dvec tfr);

  double simulationTime  = 0.0;
  double energy1         = 0.0;
  double energy2         = 0.0;
  double maxBubbleRadius = 0.0;
  size_t integrationStep = 0;
  uint32_t numSnapshots  = 0;
  int numBubbles         = 0;
  int maxNumCells        = 0;
  int numPairs           = 0;

  cudaStream_t velocityStream;
  cudaStream_t gasExchangeStream;

  KernelSize pairKernelSize;

  // Host pointers to device global variables
  int *mbpc      = nullptr;
  int *np        = nullptr;
  double *dtfapr = nullptr;
  double *dtfa   = nullptr;
  double *dvm    = nullptr;
  double *dtv    = nullptr;
  double *dir    = nullptr;
  double *dta    = nullptr;
  double *dasai  = nullptr;

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

  double *deviceDoubles = nullptr;
  uint32_t dataStride   = 0;

  // Device double pointers
  enum class DDP
  {
    X,
    Y,
    Z,
    R,

    DXDT,
    DYDT,
    DZDT,
    DRDT,

    DXDTO,
    DYDTO,
    DZDTO,
    DRDTO,

    X0,
    Y0,
    Z0,

    PATH,
    DISTANCE,

    XP,
    YP,
    ZP,
    RP,

    DXDTP,
    DYDTP,
    DZDTP,
    DRDTP,

    ERROR,

    TEMP1,
    TEMP2,
    TEMP3,
    TEMP4,
    TEMP5,
    TEMP6,
    TEMP7,
    TEMP8,

    NUM_VALUES
  };

  std::array<double *, (uint64_t)DDP::NUM_VALUES> ddps;
};

enum class CellProperty
{
  OFFSET,
  SIZE,

  NUM_VALUES
};
}; // namespace cubble
