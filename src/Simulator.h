// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "Vec.h"
#include "CudaContainer.h"
#include "Cell.h"
#include "DeviceMemoryHandler.h"

#include <cuda_runtime.h>
#include <memory>

namespace cubble
{
    class Simulator
    {
	CUBBLE_PROP(double, SimulationTime, 0)
	CUBBLE_PROP(double, ElasticEnergy, 0)
    public:
        Simulator(std::shared_ptr<Env> e);
	~Simulator();

	void setupSimulation();
	void integrate(bool useGasExchange = false);
	double getVolumeOfBubbles() const;
	double getAverageRadius() const;
	void getBubbles(std::vector<Bubble> &bubbles) const;
	
    private:
	template<typename T, typename InputIterT, typename OutputIterT>
        T cubReduction(cudaError_t (* reduxFunc)(void*, size_t&, InputIterT, OutputIterT, int, cudaStream_t, bool),
		       InputIterT deviceInputData,
		       size_t numValues) const
	{
	    assert(deviceInputData != nullptr);

	    OutputIterT deviceOutputData = static_cast<OutputIterT>(
		dmh->getDataPtr(TemporaryBubbleProperty::REDUCTION_OUTPUT));

	    size_t tempStorageBytes = 0;
	    void *devTempStoragePtr = NULL;
	    (*reduxFunc)(devTempStoragePtr, tempStorageBytes, deviceInputData, deviceOutputData, numValues, 0, false);
	    assert(tempStorageBytes <=
		   dmh->getNumBytesOfMemoryFromPropertyToEnd(TemporaryBubbleProperty::REDUCTION_TEMP));
	    devTempStoragePtr = static_cast<void*>(dmh->getDataPtr(TemporaryBubbleProperty::REDUCTION_TEMP));

	    (*reduxFunc)(devTempStoragePtr, tempStorageBytes, deviceInputData, deviceOutputData, numValues, 0, false);
	    T hostOutputData;
	    cudaMemcpy(&hostOutputData, deviceOutputData, sizeof(T), cudaMemcpyDeviceToHost);

	    return hostOutputData;
	}
	    
	void generateBubbles();
	void assignBubblesToCells(bool useVerboseOutput = false);
	dim3 getGridSize();

	size_t givenNumBubblesPerDim = 0;
	size_t numBubbles = 0;
	const int neighborStride = 30;
	size_t integrationStep = 0;

	cudaEvent_t start = 0;
	cudaEvent_t stop = 0;

	std::unique_ptr<DeviceMemoryHandler> dmh;
	std::shared_ptr<Env> env;
        
	CudaContainer<Cell> cells;
	
	CudaContainer<int> indices;
	CudaContainer<int> numberOfNeighbors;
	CudaContainer<int> neighborIndices;
	
	CudaContainer<int> indicesToKeep;
	CudaContainer<int> numBubblesToKeep;
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
			  Cell *cells,
			  dvec domainDim,
			  int numBubbles,
			  int numCells);

    __global__
    void bubblesToCells(double *x,
			double *y,
			double *z,
			int *indices,
			Cell *cells,
			dvec domainDim,
			int numBubbles);

    __global__
    void findNeighbors(double *x,
		       double *y,
		       double *z,
		       double *r,
		       int *indices,
		       Cell *cells,
		       int *numberOfNeighbors,
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
				 
				 int *numberOfNeighbors,
				 int *neighborIndices,
				 dvec interval,
				 int numBubbles,
				 int neighborStride,
				 double pi,
				 bool useGasExchange);

    __global__
    void calculateVelocityFromAccelerations(double *ax,
					    double *ay,
					    double *az,
					    double *ar,
					    double *e,
					    
					    double *dxdt,
					    double *dydt,
					    double *dzdt,
					    double *drdt,
					    
					    double *energies,
					    
					    int numBubbles,
					    int neighborStride,
					    double fZeroPerMuZero,
					    double kParam);
    
    __global__
    void accelerate(double *x,
		    double *y,
		    double *z,
		    double *r,
		    
		    double *dxdt,
		    double *dydt,
		    double *dzdt,
		    double *drdt,
		    
		    double *energies,
		    int *numberOfNeighbors,
		    int *neighborIndices,
		    dvec tfr,
		    dvec lbb,
		    int numBubbles,
		    int neighborStride,
		    double fZeroPerMuZero,
		    double kParam,
		    double pi,
		    double minRad,
		    bool useGasExchange);

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
    void updateData(double *x,
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
		    
		    double *dxdtPrd,
		    double *dydtPrd,
		    double *dzdtPrd,
		    double *drdtPrd,

		    double *volumes,
		    int *numBubblesToKeep,
		    int *indicesToKeep,
		    int numBubbles,
		    double minRad,
		    double pi,
		    bool useGasExchange);

    __global__
    void removeSmallBubbles(double *currentData,
			    double *temporaryData,
			    int *indicesToKeep,
			    int numBubblesToKeep,
			    int numBubbles,
			    int memoryStride,
			    int memoryIndexOfRadius,
			    double invPi,
			    double deltaVolume);

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
				      Cell *cells,
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
