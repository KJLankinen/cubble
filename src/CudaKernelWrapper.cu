// -*- C++ -*-

#include "CudaKernelWrapper.h"
#include "Macros.h"
#include "CudaContainer.h"

#include <iostream>
#include <curand.h>

cubble::CudaKernelWrapper::CudaKernelWrapper(std::shared_ptr<BubbleManager> bm,
					     std::shared_ptr<Env> e)
{
    bubbleManager = bm;
    env = e;
}

cubble::CudaKernelWrapper::~CudaKernelWrapper()
{}

void cubble::CudaKernelWrapper::generateBubblesOnGPU()
{
    // Get necessary parameters
    int n = env->getNumBubbles();
    int numBlocksPerDim = env->getNumCellsPerDim();
    int rngSeed = env->getRngSeed();
    double avgRad = env->getAvgRad();
    double stdDevRad = env->getStdDevRad();
    dvec lbb = env->getLbb();
    dvec tfr = env->getTfr();
	
    int totalNumBlocks = numBlocksPerDim * numBlocksPerDim;
#if (NUM_DIM == 3)
    totalNumBlocks *= numBlocksPerDim;
    int numThreadsPerDim = (int)std::ceil(std::cbrt(n / (float)totalNumBlocks));
    dim3 blockSize = dim3(numThreadsPerDim * numThreadsPerDim * numThreadsPerDim, 1, 1);
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim, numBlocksPerDim);
#else
    int numThreadsPerDim = (int)std::ceil(std::sqrt(n / (float)totalNumBlocks));
    dim3 blockSize = dim3(numThreadsPerDim * numThreadsPerDim, 1, 1);
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim, 1);
#endif
    int minNumCells = (int)std::ceil(n / 1024.0f);
    int numCells = gridSize.x * gridSize.y * gridSize.z;
    int numThreads = blockSize.x * blockSize.y * blockSize.z;
    if (numThreads > 1024)
    {
	std::cerr << "Too many bubbles to simulate w.r.t. the number of blocks."
		  << "\nAmount of threads per block " << numThreads
		  << " exceeds the maximum, which is 1024."
		  << "\nIncrease the number of blocks (cells) per dimension."
		  << "\nMinimum (total) number of cells with given number of bubbles: "
		  << minNumCells
		  << std::endl;
	
	std::exit(EXIT_FAILURE);
    }

    std::cout << "Grid size: (" << gridSize.x
	      << ", " << gridSize.y
	      << ", " << gridSize.z
	      << "), block size: (" << blockSize.x
	      << ", " << blockSize.y
	      << ", " << blockSize.z
	      << ")\nMinimum (total) number of cells with given number of bubbles: "
	      << minNumCells
	      << std::endl;
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
#endif
    CudaContainer<float> r(n);
    CudaContainer<Bubble> b(n);
    CudaContainer<int> bubbleIndices(n);
    CudaContainer<int> bubblesPerCell(numCells);
    CudaContainer<int> offsets(numCells);

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    CURAND_CALL(curandGenerateUniform(generator, x.getDevicePtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDevicePtr(), n));
    CURAND_CALL(curandGenerateNormal(generator, r.getDevicePtr(), n, avgRad, stdDevRad));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniform(generator, z.getDevicePtr(), n));
#endif

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
    bubblesPerCell.toHost();
    bubbleManager->cellEnds.resize(numCells);
    bubbleManager->cellBegins.resize(numCells);

    // Bubbles are stored in a 1D array, even if the domain is 2D or 3D.
    // Here we calculate the offsets (begin and end indices) to the memory
    // location where the bubbles of the current cell start (and end).
    int offset = 0;
    for (size_t i = 0; i < numCells; ++i)
    {
	offsets[i] = offset;
	bubbleManager->cellBegins[i] = offset;
	offset += bubblesPerCell[i];
	bubbleManager->cellEnds[i] = offset;
    }
    offsets.toDevice();

    // Reset bubblesPerCell to 0, and reuse it.
    bubblesPerCell.fillHostWith(0);
    bubblesPerCell.toDevice();

    assignBubblesToCells<<<gridSize, blockSize>>>(b.getDevicePtr(),
						  bubbleIndices.getDevicePtr(),
						  offsets.getDevicePtr(),
						  bubblesPerCell.getDevicePtr(),
						  lbb,
						  tfr,
						  n);
    bubbleIndices.toHost();
    b.toHost();

    // Create a new host vector and add the bubbles in to it
    // s.t. the bubbles of any one cell are continuously stored.
    // Note that the memory locality could probably be improved
    // for nearby cells by saving the cells as a Hilbert curve,
    // instead of left to right, bottom to top, back to front.
    std::vector<Bubble> bubbles;
    bubbles.resize(n);
    for (size_t i = 0; i < n; ++i)
	bubbles[i] = b[bubbleIndices[i]];

    bubbleManager->setBubbles(bubbles);
}

__forceinline__ __device__
int getTid()
{
    // Simple helper function for calculating a 1D coordinate
    // from 1, 2 or 3 dimensional coordinates.
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int blocksBefore = blockIdx.z * (gridDim.y * gridDim.x)
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;
    int threadsBefore = blockDim.y * blockDim.x * threadIdx.z + blockDim.x * threadIdx.y;
    int tid = blocksBefore * threadsPerBlock + threadsBefore + threadIdx.x;

    return tid;
}

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
    int tid = getTid();
    if (tid < numBubbles)
    {
	dvec pos;
        pos.x = (double)x[tid];
	pos.y = (double)y[tid];
	
	int bbi = ((int)(pos.y * gridDim.y) * gridDim.x) + (int)(pos.x * gridDim.x);
	
#if (NUM_DIM == 3)
	pos.z = (double)z[tid];
	bbi += ((int)(pos.z * gridDim.z) * gridDim.y * gridDim.x);
#endif
	// Scale position
	pos = pos * tfr + lbb;

	b[tid].setPos(pos);
	b[tid].setRadius((double)r[tid]);
	
        atomicAdd(&bubblesPerCell[bbi], 1);
    }
}

__global__
void cubble::assignBubblesToCells(Bubble *b,
				  int *bubbleIndices,
				  int *offsets,
				  int *currentIndices,
				  dvec lbb,
				  dvec tfr,
				  int numBubbles)
{
    int tid = getTid();
    dvec interval = tfr - lbb;
    dvec normPos = (b[tid].getPos() - lbb) / interval;

    if (tid < numBubbles)
    {
	int bbi = ((int)(normPos.y * gridDim.y) * gridDim.x) + (int)(normPos.x * gridDim.x);
#if (NUM_DIM == 3)
	bbi += (int)(normPos.z * gridDim.z) * gridDim.y * gridDim.x;
#endif
	int offset = atomicAdd(&currentIndices[bbi], 1) + offsets[bbi];
	bubbleIndices[offset] = tid;
    }
}
