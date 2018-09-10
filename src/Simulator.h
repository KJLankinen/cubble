// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "Vec.h"
#include "FixedSizeDeviceArray.h"

#include <cuda_runtime.h>
#include <nvToolsExt.h>
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

	ENERGY,
	FREE_AREA,
	
	DXDT_OLD,
	DYDT_OLD,
	DZDT_OLD,
	DRDT_OLD,
	
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
	template<typename T, typename InputIterT, typename OutputIterT>
        T cubReduction(cudaError_t (* func)(void*, size_t&, InputIterT, OutputIterT, int, cudaStream_t, bool),
		       InputIterT deviceInputData,
		       int numValues)
	{
	    NVTX_RANGE_PUSH_A("CubReductionTemplateFunc");
	    assert(deviceInputData != nullptr);

	    NVTX_RANGE_PUSH_A("RedOutputDataCheck");
	    if (sizeof(T) > cubOutputData.getSizeInBytes())
	        cubOutputData = FixedSizeDeviceArray<char>(sizeof(T), 1);
	    NVTX_RANGE_POP();

	    
	    NVTX_RANGE_PUSH_A("StaticCast");
	    void *rawOutputPtr = static_cast<void*>(cubOutputData.getDataPtr());
	    OutputIterT deviceOutputData = static_cast<OutputIterT>(rawOutputPtr);
	    NVTX_RANGE_POP();


	    NVTX_RANGE_PUSH_A("TempSizeFetch");
	    size_t tempStorageBytes = 0;
	    (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, 0, false);
	    NVTX_RANGE_POP();


	    NVTX_RANGE_PUSH_A("TempResize");
	    if (tempStorageBytes > cubTemporaryStorage.getSizeInBytes())
		cubTemporaryStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);
	    NVTX_RANGE_POP();


	    NVTX_RANGE_PUSH_A("Reduce");
	    void *tempStoragePtr = static_cast<void*>(cubTemporaryStorage.getDataPtr());
	    (*func)(tempStoragePtr,
		    tempStorageBytes,
		    deviceInputData,
		    deviceOutputData,
		    numValues,
		    0,
		    false);
	    NVTX_RANGE_POP();


    	    NVTX_RANGE_PUSH_A("MemcpyBackToHost");
	    T hostOutputData;
	    cudaMemcpyAsync(&hostOutputData, deviceOutputData, sizeof(T), cudaMemcpyDeviceToHost);
	    NVTX_RANGE_POP();

	    NVTX_RANGE_POP();

	    return hostOutputData;
	}

	template<typename InputIterT, typename OutputIterT>
	void cubScan(cudaError_t (* func)(void*, size_t&, InputIterT, OutputIterT, int, cudaStream_t, bool),
		     InputIterT deviceInputData,
		     OutputIterT deviceOutputData,
		     int numValues)
	{
	    assert(deviceInputData != nullptr);
	    assert(deviceOutputData != nullptr);

	    size_t tempStorageBytes = 0;
	    (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, 0, false);

	    if (tempStorageBytes > cubTemporaryStorage.getSizeInBytes())
		cubTemporaryStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);
	 
	    void *tempStoragePtr = static_cast<void*>(cubTemporaryStorage.getDataPtr());
	    (*func)(tempStoragePtr,
		    tempStorageBytes,
		    deviceInputData,
		    deviceOutputData,
		    numValues,
		    0,
		    false);
	}

	template<typename KeyT, typename ValueT>
	void cubSortPairs(cudaError_t (*func)(
			      void*,
			      size_t&,
			      const KeyT*,
			      KeyT*,
			      const ValueT*,
			      ValueT*,
			      int,
			      int,
			      int,
			      cudaStream_t,
			      bool),
			  const KeyT *keysIn,
			  KeyT *keysOut,
			  const ValueT *valuesIn,
			  ValueT *valuesOut,
			  int numValues)
	{
	    assert(keysIn != nullptr);
	    assert(keysOut != nullptr);
	    assert(valuesIn != nullptr);
	    assert(valuesOut != nullptr);

	    size_t tempStorageBytes = 0;
	    (*func)(NULL,
		    tempStorageBytes,
		    keysIn,
		    keysOut,
		    valuesIn,
		    valuesOut,
		    numValues,
		    0,
		    sizeof(KeyT) * 8,
		    0,
		    false);

	    if (tempStorageBytes > cubTemporaryStorage.getSizeInBytes())
		cubTemporaryStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);
	    
	    void *tempStoragePtr = static_cast<void*>(cubTemporaryStorage.getDataPtr());
	    (*func)(tempStoragePtr,
		    tempStorageBytes,
		    keysIn,
		    keysOut,
		    valuesIn,
		    valuesOut,
		    numValues,
		    0,
		    sizeof(KeyT) * 8,
		    0,
		    false);
	}
   
	void generateBubbles();
	void updateCellsAndNeighbors();
	void updateData();
        bool deleteSmallBubbles();
	dim3 getGridSize();

	size_t givenNumBubblesPerDim = 0;
	size_t numBubbles = 0;
        int hostNumPairs = 0;

	const static int neighborStride = 32;
	static_assert(neighborStride % 4 == 0, "Neigbor stride must be divisible by 4.");

	size_t integrationStep = 0;

	cudaEvent_t start = 0;
	cudaEvent_t stop = 0;

	std::shared_ptr<Env> env;

	FixedSizeDeviceArray<double> bubbleData;
        FixedSizeDeviceArray<int> aboveMinRadFlags;
	FixedSizeDeviceArray<int> cellData;
	FixedSizeDeviceArray<int> indicesPerCell;
        FixedSizeDeviceArray<int> neighborPairIndices;
	FixedSizeDeviceArray<int> numPairs;
	
	FixedSizeDeviceArray<char> cubOutputData;
	FixedSizeDeviceArray<char> cubTemporaryStorage;
        
	std::vector<double> hostData;
    };
    
    // ******************************
    // Kernels
    // ******************************

    __global__
    void calculateVolumes(double *r, double *volumes, int numBubbles, double pi);
    
    __global__
    void assignDataToBubbles(double *x,
			     double *y,
			     double *z,
			     double *xPrd,
			     double *yPrd,
			     double *zPrd,
			     double *r,
			     double *w,
			     int *aboveMinRadFlags,
			     int givenNumBubblesPerDim,
			     dvec tfr,
			     dvec lbb,
			     double avgRad,
			     double minRad,
			     int numBubbles);

    __global__
    void calculateOffsets(double *x,
			  double *y,
			  double *z,
			  int *sizes,
			  dvec domainDim,
			  int numBubbles,
			  int numCells);

    __global__
    void bubblesToCells(double *x,
			double *y,
			double *z,
			int *indices,
		        int *offsets,
			int *sizes,
			dvec domainDim,
			int numBubbles);

    __global__
    void findBubblePairs(double *x,
			 double *y,
			 double *z,
			 double *r,
			 int *indices,
			 int *offsets,
			 int *sizes,
			 int *firstIndices,
			 int *secondIndices,
			 int *numPairs,
			 int numCells,
			 int numBubbles,
			 dvec interval,
			 int maxNumSharedVals,
			 int maxNumPairs);
    
    __global__
    void predict(double *x,
		 double *y,
		 double *z,
		 double *r,
		 
		 double *xPrd,
		 double *yPrd,
		 double *zPrd,
		 double *rPrd,
		 
		 double *dxdt,
		 double *dydt,
		 double *dzdt,
		 double *drdt,
		 
		 double *dxdtOld,
		 double *dydtOld,
		 double *dzdtOld,
		 double *drdtOld,
		 
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 bool useGasExchange);

    __global__
    void calculateVelocityAndGasExchange(double *x,
					 double *y,
					 double *z,
					 double *r,
					 
					 double *dxdt,
					 double *dydt,
					 double *dzdt,
					 double *drdt,
					 
					 double *energy,
					 double *freeArea,
					 
					 int *firstIndices,
					 int *secondIndices,
					 
					 int numBubbles,
					 int numPairs,
					 double fZeroPerMuZero,
					 double pi,
					 dvec interval,
					 bool calculateEnergy,
					 bool useGasExchange);

    __global__
    void calculateFreeAreaPerRadius(double *r, double *freeArea, double *output, double pi, int numBubbles);

    __global__
    void calculateFinalRadiusChangeRate(double *drdt,
					double *r,
					double *freeArea,
					int numBubbles,
					double invRho,
					double invPi,
					double kappa,
					double kParam);
    
    __global__
    void correct(double *x,
		 double *y,
		 double *z,
		 double *r,
		 
		 double *xPrd,
		 double *yPrd,
		 double *zPrd,
		 double *rPrd,
		 
		 double *dxdt,
		 double *dydt,
		 double *dzdt,
		 double *drdt,
		 
		 double *dxdtPrd,
		 double *dydtPrd,
		 double *dzdtPrd,
		 double *drdtPrd,
		 
		 double *errors,
		 int *aboveMinRadFlags,
		 double minRad,
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 bool useGasExchange);

    __global__
    void addVolume(double *r, double *volumeMultiplier, int numBubbles, double invTotalVolume);

    __global__
    void eulerIntegration(double *x,
			  double *y,
			  double *z,
			  double *r,
			  
			  double *dxdt,
			  double *dydt,
			  double *dzdt,
			  double *drdt,
			  
			  dvec tfr,
			  dvec lbb,
			  double timeStep,
			  int numBubbles);

    __global__
    void calculateRedistributedGasVolume(double *volume,
					 double *r,
					 int *aboveMinRadFlags,
					 double *volumeMultiplier,
					 double pi,
					 int numBubbles);

    __global__
    void removeSmallBubbles(double *x,
			    double *y,
			    double *z,
			    double *r,
			    
			    double *xTemp,
			    double *yTemp,
			    double *zTemp,
			    double *rTemp,
			    
			    double *dxdt,
			    double *dydt,
			    double *dzdt,
			    double *drdt,
			    
			    double *dxdtTemp,
			    double *dydtTemp,
			    double *dzdtTemp,
			    double *drdtTemp,
			    
			    double *dxdtOld,
			    double *dydtOld,
			    double *dzdtOld,
			    double *drdtOld,
			    
			    double *dxdtOldTemp,
			    double *dydtOldTemp,
			    double *dzdtOldTemp,
			    double *drdtOldTemp,
			    
			    int *newIdx,
			    int *flag,
			    int numBubbles);
    

    // ******************************
    // Device functions
    // ******************************

    __device__
    double getWrappedCoordinate(double val1, double val2, double multiplier);
    
    __device__
    int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
    
    __device__
    int getGlobalTid();

    __device__
    dvec getWrappedPos(dvec pos);
};
