// -*- C++ -*-

#include "CudaKernelWrapper.h"
#include "Macros.h"
#include "CudaContainer.h"

#include <iostream>
#include <curand.h>

cubble::CudaKernelWrapper::CudaKernelWrapper(std::shared_ptr<BubbleManager> bm)
{
    bubbleManager = bm;
}

cubble::CudaKernelWrapper::~CudaKernelWrapper()
{}

void cubble::CudaKernelWrapper::generateBubblesOnGPU(size_t n,
						     size_t numBlocksPerDim,
						     int rngSeed,
						     double avgRad,
						     double stdDevRad,
						     dvec lbb,
						     dvec tfr)
{
#if (NUM_DIM == 3)
    size_t blockSize = std::ceil(n / (float)(numBlocksPerDim
					     * numBlocksPerDim
					     * numBlocksPerDim));
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim, numBlocksPerDim);
    size_t recommendedNumCells = (size_t)(std::ceil(std::cbrt(n / 1024.0f) / 8)) * 8;
#else
    size_t blockSize = std::ceil(n / (float)(numBlocksPerDim * numBlocksPerDim));
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim);
    size_t recommendedNumCells = (size_t)(std::ceil(std::sqrt(n / 1024.0f) / 8)) * 8;
#endif

    if (blockSize > 1024)
    {
	std::cerr << "Too many bubbles to simulate w.r.t. the number of blocks."
		  << "\nAmount of threads per block " << blockSize
		  << " exceeds the maximum, which is 1024."
		  << "\nIncrease the number of blocks (cells) per dimension."
		  << "\nRecommended number of cells with given number of bubbles: "
		  << recommendedNumCells
		  << std::endl;
	
	std::exit(EXIT_FAILURE);
    }

    std::cout << "Grid size: " << gridSize.x << ", " << gridSize.y << ", " << gridSize.z
	      << ", block size: " << blockSize
	      << "\nRecommended number of cells with given number of bubbles: "
	      << recommendedNumCells
	      << std::endl;
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
#endif
    
    CudaContainer<float> r(n);
    CudaContainer<Bubble> b(n);
    CudaContainer<int> bubbleIndices(n);

    CudaContainer<int> bubblesPerCell(gridSize.x * gridSize.y * gridSize.z);
    CudaContainer<int> offsets(gridSize.x * gridSize.y * gridSize.z);

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    
    CURAND_CALL(curandGenerateUniform(generator, x.getDevicePtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDevicePtr(), n));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniform(generator, z.getDevicePtr(), n));
#endif
    CURAND_CALL(curandGenerateNormal(generator, r.getDevicePtr(), n, avgRad, stdDevRad));

    // Assign generated data to bubbles
    assignDataToBubbles<<<gridSize, blockSize>>>(x.getDevicePtr(),
						 y.getDevicePtr(),
#if (NUM_DIM == 3)
						 z.getDevicePtr(),
#endif
						 r.getDevicePtr(),
						 b.getDevicePtr(),
						 bubblesPerCell.getDevicePtr(),
						 lbb,
						 tfr,
						 n);

    // Assign bubbles to cells
    // For this, compute offsets to memory based on how many bubbles are in each cell.
    // Then reset bubblesPerCell back to zero so it can be used to index the bubbleIndices
    // mem loc again.
    bubblesPerCell.toHost();
    size_t offset = 0;
    for (size_t i = 0; i < gridSize.x * gridSize.y * gridSize.z; ++i)
    {
	offsets[i] = offset;
	offset += bubblesPerCell[i];
    }
    offsets.toDevice();
    bubblesPerCell.fillHostWith(0);
    bubblesPerCell.toDevice();

    assignBubblesToCells<<<gridSize, blockSize>>>(b.getDevicePtr(),
						  bubbleIndices.getDevicePtr(),
						  offsets.getDevicePtr(),
						  bubblesPerCell.getDevicePtr(),
						  n);
    bubbleIndices.toHost();
    b.toHost();

    std::vector<Bubble> bubbles;
    bubbles.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
	// Add the bubbles to a new vector, ordered by cell
	bubbles[i] = b[bubbleIndices[i]];
	
	// Update the index of the bubble
	bubbleIndices[i] = i;
    }
    
    bubbleManager->bubbles = bubbles;
    bubbleIndices.copyHostDataToVec(bubbleManager->indices);
}

// ----------------------
// Kernel implementations
// ----------------------

__global__
void cubble::assignDataToBubbles(float *x,
				 float *y,
#if (NUM_DIM == 3)
				 float *z,
#endif
				 float *r,
				 Bubble *b,
				 int *bubblesPerCell,
				 dvec lbb,
				 dvec tfr,
				 int numBubbles)
{
    int bbi = 0;
    
#if (NUM_DIM == 3)
    
    // 3D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.z * (gridDim.y * gridDim.x)
		  + blockIdx.y * gridDim.x + blockIdx.x)
	* blockDim.x + threadIdx.x;
    
    // Block index of bubble changed to 1D.
    if (tid < numBubbles)
	bbi = ((int)(z[tid] * gridDim.z) * gridDim.y * gridDim.x)
	    + ((int)(y[tid] * gridDim.y) * gridDim.x)
	    + (int)(x[tid] * gridDim.x);
#else
    
    // 2D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    
    // Block index of bubble changed to 1D.
    if (tid < numBubbles)
	bbi = ((int)(y[tid] * gridDim.y) * gridDim.x) + (int)(x[tid] * gridDim.x);
    
#endif
    
    if (tid < numBubbles)
    {
        int indexToCellContainer = atomicAdd(&bubblesPerCell[bbi], 1);
	
	b[tid].pos.x = lbb.x + (double)x[tid] * tfr.x;
	b[tid].pos.y = lbb.y + (double)y[tid] * tfr.y;
#if (NUM_DIM == 3)
	b[tid].pos.z = lbb.z + (double)z[tid] * tfr.z;	
#endif
	b[tid].radius = (double)r[tid];
    }
}

__global__
void cubble::assignBubblesToCells(Bubble *b,
				  int *bubbleIndices,
				  int *offsets,
				  int *currentIndices,
				  int numBubbles)
{
    int bbi = 0;
    
#if (NUM_DIM == 3)
    
    // 3D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.z * (gridDim.y * gridDim.x)
		  + blockIdx.y * gridDim.x + blockIdx.x)
	* blockDim.x + threadIdx.x;
    
    // Block index of bubble changed to 1D.
    if (tid < numBubbles)
	bbi = ((int)(b[tid].pos.z * gridDim.z) * gridDim.y * gridDim.x)
	    + ((int)(b[tid].pos.y * gridDim.y) * gridDim.x)
	    + (int)(b[tid].pos.x * gridDim.x);
#else
    // 2D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    
    // Block index of bubble changed to 1D.
    if (tid < numBubbles)
	bbi = ((int)(b[tid].pos.y * gridDim.y) * gridDim.x)
	    + (int)(b[tid].pos.x * gridDim.x);
#endif

    if (tid < numBubbles)
    {
	// Save the index of the bubble to the array.
	// Offsets array stores an offset to the continuous array,
	// where the indices of the bubbles belonging to this cell start from.
	// atomicAdd is used to get the next unused index of the cell bbi.
	
	int offset = atomicAdd(&currentIndices[bbi], 1) + offsets[bbi];
	bubbleIndices[offset] = tid;
    }
}
