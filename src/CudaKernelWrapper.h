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

	void generateBubblesOnGPU();
	
    private:
	__device__
	int getTid();
	
	std::shared_ptr<BubbleManager> bubbleManager;
	std::shared_ptr<Env> env;
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
			      Bubble *reorganizedBubbles,
			      int *memoryOffsets,
			      int *currendIndices,
			      dvec lbb,
			      dvec tfr,
			      int numBubbles);
};
