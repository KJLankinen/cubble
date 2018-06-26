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

    __global__
    void calculateVolumes(Bubble *b, double *volumes, int numBubbles);
    
    __device__
    int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);

    __device__
    void getDomainOffsetsAndIntervals(int numBubbles,
				      int numDomains,
				      int numCells,
				      int numLocalBubbles,
				      ivec cellIdxVec,
				      ivec boxDim,
				      Cell *cells,
				      int &outXBegin,
				      int &outXInterval,
				      int &outYBegin,
				      int &outYInterval);
    
    __device__
    int getGlobalTid();

    __device__
    double getWrappedSquaredLength(dvec tfr, dvec lbb, dvec pos1, dvec pos2);

    __global__
    void assignDataToBubbles(float *x,
			     float *y,
#if (NUM_DIM == 3)
			     float *z,
#endif
			     float *r,
			     Bubble *b,
			     dvec lbb,
			     dvec tfr,
			     int numBubbles);

    __global__
    void calculateOffsets(Bubble *bubbles,
			  Cell *cells,
			  dvec lbb,
			  dvec tfr,
			  int numBubbles);

    __global__
    void bubblesToCells(Bubble *bubbles,
			int *indices,
			Cell *cells,
			int numBubbles);

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
};
