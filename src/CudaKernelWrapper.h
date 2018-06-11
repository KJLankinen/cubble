// -*- C++ -*-

#pragma once

#include "Env.h"
#include "Bubble.h"
#include "BubbleManager.h"
#include "Vec.h"

#include <cuda_runtime.h>
#include <memory>

namespace cubble
{
    class CudaKernelWrapper
    {
    public:
        CudaKernelWrapper(std::shared_ptr<BubbleManager> bm, std::shared_ptr<Env> e);
	~CudaKernelWrapper();

	void generateBubbles(std::vector<Bubble> &outBubbles);
	void assignBubblesToCells(const std::vector<Bubble> &b);
	void removeIntersectingBubbles(const std::vector<Bubble> &b);
	
    private:

	dim3 getGridSize(int numBubbles);
	
	std::shared_ptr<BubbleManager> bubbleManager;
	std::shared_ptr<Env> env;
    };
    
    __device__
    int getGlobalTid();

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
    void assignBubblesToCells(Bubble *bubbles,
			      int *indices,
			      Cell *cells,
			      int numBubbles);
};
