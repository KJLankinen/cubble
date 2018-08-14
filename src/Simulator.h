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
	double sumReduction(double *inputData, size_t lengthOfData) const;
	double getMaxElement(double *inputData, size_t lengthOfData) const;
	void generateBubbles();
	void assignBubblesToCells(bool useVerboseOutput = false);
	dim3 getGridSize();

	size_t givenNumBubblesPerDim = 0;
	size_t numBubbles = 0;
	const int neighborStride = 100;
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
