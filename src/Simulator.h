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
    public:
        Simulator(std::shared_ptr<Env> e);
	~Simulator();

	void setupSimulation();
	void integrate();
	double getVolumeOfBubbles() const;
	void getBubbles(std::vector<Bubble> &b) const;
	
    private:
	dim3 getGridSize(int numBubbles);
	void generateBubbles();
	void assignBubblesToCells();
	void removeIntersectingBubbles();
        
	std::shared_ptr<Env> env;
	CudaContainer<Bubble> bubbles;
	CudaContainer<int> indices;
	CudaContainer<Cell> cells;
    };
    
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
    
    __global__
    void calculateVolumes(Bubble *b, double *volumes, int numBubbles);
    
    __global__
    void assignDataToBubbles(float *x,
			     float *y,
#if (NUM_DIM == 3)
			     float *z,
#endif
			     float *r,
			     Bubble *b,
			     int numBubbles);

    __global__
    void calculateOffsets(Bubble *bubbles, Cell *cells, int numBubbles);

    __global__
    void bubblesToCells(Bubble *bubbles, int *indices, Cell *cells, int numBubbles);

    __global__
    void findIntersections(Bubble *bubbles,
			   int *indices,
			   Cell *cells,
			   int *intesectingIndices,
			   dvec tfr,
			   dvec lbb,
			   double overlapTolerance,
			   int numBubbles,
			   int numDomains,
			   int numCells,
			   int numLocalBubbles);

    __global__
    void predict(Bubble *bubbles,
		 int *indices,
		 Cell *cells,
		 dvec tfr,
		 dvec lbb,
		 double timeStep,
		 int numBubbles,
		 int numCells);

    __global__
    void correct(Bubble *bubbles,
		 int *indices,
		 Cell *cells,
		 double *errors,
		 dvec tfr,
		 dvec lbb,
		 double fZeroPerMuZero,
		 double timeStep,
		 int numBubbles,
		 int numCells);
    
    __global__
    void accelerate(Bubble *bubbles,
		    int *indices,
		    Cell *cells,
		    dvec tfr,
		    dvec lbb,
		    int numBubbles,
		    int numDomains,
		    int numCells,
		    int numLocalBubbles);
};
