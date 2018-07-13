// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "Vec.h"
#include "CudaContainer.h"
#include "Cell.h"

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
	void integrate(bool useGasExchange = false, bool printTimings = false);
	double getVolumeOfBubbles() const;
	void getBubbles(std::vector<Bubble> &b) const;
	void assignBubblesToCells(bool useVerboseOutput = false);
	
    private:
	dim3 getGridSize(int numBubbles);
	void generateBubbles();

	const int neighborStride = 100;
	size_t integrationStep = 0;

	cudaEvent_t start = 0;
	cudaEvent_t stop = 0;
	
	std::shared_ptr<Env> env;
	CudaContainer<Bubble> bubbles;
	CudaContainer<int> indices;
	CudaContainer<Cell> cells;
	CudaContainer<double> errors;
	CudaContainer<double> energies;
	CudaContainer<int> numberOfNeighbors;
	CudaContainer<int> neighborIndices;
    };

    
    // ******************************
    // Kernels
    // ******************************

    __global__
    void getRadii(Bubble *b, double *radii, int numBubbles);
    
    __global__
    void calculateVolumes(Bubble *b, double *volumes, int numBubbles, double pi);
    
    __global__
    void assignDataToBubbles(float *x,
			     float *y,
#if (NUM_DIM == 3)
			     float *z,
#endif
			     float *r,
			     float *w,
			     Bubble *b,
			     ivec bubblesPerDim,
			     dvec tfr,
			     dvec lbb,
			     double avgRad,
			     int numBubbles);

    __global__
    void calculateOffsets(Bubble *bubbles, Cell *cells, int numBubbles, int numCells);

    __global__
    void bubblesToCells(Bubble *bubbles, int *indices, Cell *cells, int numBubbles);

    __global__
    void findNeighbors(Bubble *bubbles,
		       int *indices,
		       Cell *cells,
		       int *numberOfNeighbors,
		       int *neighborIndices,
		       dvec tfr,
		       dvec lbb,
		       int numBubbles,
		       int numDomains,
		       int numCells,
		       int numLocalBubbles,
		       int neighborStride);
    
    __global__
    void predict(Bubble *bubbles,
		 int *indices,
		 Cell *cells,
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 int numCells,
		 bool useGasExchange);

    __global__
    void correct(Bubble *bubbles,
		 int *indices,
		 Cell *cells,
		 double *errors,
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 int numCells,
		 bool useGasExchange);
    
    __global__
    void accelerate(Bubble *bubbles,
		    int *indices,
		    Cell *cells,
		    int *numberOfNeighbors,
		    int *neighborIndices,
		    double *energies,
		    dvec tfr,
		    dvec lbb,
		    int numBubbles,
		    int numCells,
		    int neighborStride,
		    double fZeroPerMuZero,
		    double kParam,
		    double pi,
		    double minRad,
		    bool useGasExchange);

    __global__
    void updateData(Bubble* bubbles,
		    int *indices,
		    Cell *cells,
		    int *toBeDeletedIndices,
		    int *numBubblesToDelete,
		    int numBubbles,
		    int numCells,
		    double minRad,
		    bool useGasExchange);


    // ******************************
    // Device functions
    // ******************************
    
    __device__
    double atomicAddD(double *address, double val);
    
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
