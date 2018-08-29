// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "Vec.h"
#include "FixedSizeDeviceArray.h"

#include <cuda_runtime.h>
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
	ERROR,
	VOLUME,
	FREE_AREA,
	
	NUM_VALUES
    };
    
    enum class BubblePairProperty
    {
	ACCELERATION_X,
	ACCELERATION_Y,
	ACCELERATION_Z,
	ACCELERATION_R,
	ENERGY,
	OVERLAP_AREA,
	
	NUM_VALUES
    };

    enum class CellProperty
    {
	OFFSET,
	SIZE,
	
	NUM_VALUES
    };

    enum class MiscIntProperty
    {
	NUM_NEIGHBORS,
	INDEX,
	
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
		       size_t numValues)
	{
	    assert(deviceInputData != nullptr);

	    if (sizeof(T) > cubOutputData.getSizeInBytes())
	        cubOutputData = FixedSizeDeviceArray<char>(sizeof(T), 1);

	    void *rawOutputPtr = static_cast<void*>(cubOutputData.getDataPtr());
	    OutputIterT deviceOutputData = static_cast<OutputIterT>(rawOutputPtr);
	    
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
	    
	    T hostOutputData;
	    cudaMemcpy(&hostOutputData, deviceOutputData, sizeof(T), cudaMemcpyDeviceToHost);

	    return hostOutputData;
	}

	template<typename InputIterT, typename OutputIterT>
	void cubScan(cudaError_t (* func)(void*, size_t&, InputIterT, OutputIterT, int, cudaStream_t, bool),
		     InputIterT deviceInputData,
		     OutputIterT deviceOutputData,
		     size_t numValues)
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
	    
	void generateBubbles();
	void updateCellsAndNeighbors();
	void updateData();
        bool deleteSmallBubbles();
	dim3 getGridSize();

	size_t givenNumBubblesPerDim = 0;
	size_t numBubbles = 0;
	const static int neighborStride = 32;
	static_assert(neighborStride % 4 == 0, "Neigbor stride must be divisible by 4.");
	size_t integrationStep = 0;

	cudaEvent_t start = 0;
	cudaEvent_t stop = 0;

	std::shared_ptr<Env> env;

	FixedSizeDeviceArray<double> bubbleData;
	FixedSizeDeviceArray<double> bubblePairData;
	FixedSizeDeviceArray<int> cellData;
	FixedSizeDeviceArray<int> miscData;
        FixedSizeDeviceArray<int> neighborIndices;
	
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
			     int givenNumBubblesPerDim,
			     dvec tfr,
			     dvec lbb,
			     double avgRad,
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
    void findNeighbors(double *x,
		       double *y,
		       double *z,
		       double *r,
		       int *indices,
		       int *offsets,
		       int *sizes,
		       int *numNeighbors,
		       int *neighborIndices,
		       dvec tfr,
		       dvec lbb,
		       int numBubbles,
		       int numDomains,
		       int numCells,
		       int neighborStride);
    
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
    void createAccelerationArray(double *x,
				 double *y,
				 double *z,
				 double *r,
				 
				 double *ax,
				 double *ay,
				 double *az,
				 double *ar,
				 double *e,
				 double *areaOverlap,
				 
				 int *numberOfNeighbors,
				 int *neighborIndices,
				 dvec interval,
				 int numBubbles,
				 int neighborStride,
				 double pi,
				 bool useGasExchange,
				 bool calculateEnergy);

    __global__
    void calculateVelocityFromAccelerations(double *ax,
					    double *ay,
					    double *az,
					    double *ar,
					    double *e,
					    double *areaOverlap,
					    
					    double *dxdt,
					    double *dydt,
					    double *dzdt,
					    double *drdt,
					    
					    double *freeArea,
					    double *energies,
					    
					    int numBubbles,
					    int neighborStride,
					    double fZeroPerMuZero,
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
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 bool useGasExchange);

    __global__
    void addVolume(double *r, int numBubbles, double volumeMultiplier);

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
    

    // ******************************
    // Device functions
    // ******************************
    
    __device__
    int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);

    __device__
    void getDomainOffsetsAndIntervals(int numBubbles,
				      int numDomains,
				      int numCells,
				      ivec cellIdxVec,
				      ivec boxDim,
				      int *offsets,
				      int *sizes,
				      int &outXBegin,
				      int &outXInterval,
				      int &outYBegin,
				      int &outYInterval,
				      bool &outIsOwnCell);
    
    __device__
    int getGlobalTid();

    __device__
    dvec getShortestWrappedNormalizedVec(dvec pos1, dvec pos2);

    __device__
    dvec getWrappedPos(dvec pos);
};
