// -*- C++ -*-

#pragma once

#include "CubWrapper.h"
#include "Env.h"
#include "Util.h"
#include "Vec.h"
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
  CubWrapper cubWrapper;

  std::vector<double> hostData;

  // Device data
  double *pinnedDoubles = nullptr;
  int *pinnedInts       = nullptr;
  double *deviceDoubles = nullptr;
  int *deviceInts       = nullptr;
  uint32_t dataStride   = 0;
  uint32_t pairStride   = 0;
  uint64_t memReqD      = 0;
  uint64_t memReqI      = 0;

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

  // Device int pointers
  enum class DIP
  {
    FLAGS,

    WRAP_COUNT_X,
    WRAP_COUNT_Y,
    WRAP_COUNT_Z,

    WRAP_COUNT_XP,
    WRAP_COUNT_YP,
    WRAP_COUNT_ZP,

    PAIR1,
    PAIR2,

    TEMP1,
    TEMP2,

    NUM_VALUES
  };

  std::array<double *, (uint64_t)DDP::NUM_VALUES> ddps;
  std::array<int *, (uint64_t)DIP::NUM_VALUES> dips;
};
}; // namespace cubble
