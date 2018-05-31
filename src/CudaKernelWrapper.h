// -*- C++ -*-

#pragma once

#include "Bubble.h"
#include "BubbleManager.h"

#include <cuda_runtime.h>
#include <memory>

namespace cubble
{
    class CudaKernelWrapper
    {
    public:
        CudaKernelWrapper(std::shared_ptr<BubbleManager> bm);
	~CudaKernelWrapper();
        
	void generateBubblesOnGPU(size_t n,
				  size_t numBlocksPerDim,
				  int rngSeed,
				  double avgRad,
				  double stdDevRad,
				  dvec lbb,
				  dvec tfr);
	
    private:
	std::shared_ptr<BubbleManager> bubbleManager;
    };

    // Kernels have to be defined outside class
    __global__
    void assignDataToBubbles(float *x,
			     float *y,
#if (NUM_DIM == 3)
			     float *z,
#endif
			     float *r,
			     Bubble *b,
			     int *bubblesPerCell,
			     dvec lbb,
			     dvec tfr,
			     int numBubbles);
    
    __global__
    void assignBubblesToCells(Bubble *b,
			      int *bubbleIndices,
			      int *memoryOffsets,
			      int *currendIndices,
			      int numBubbles);
};
