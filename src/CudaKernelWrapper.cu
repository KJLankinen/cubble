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

    printRelevantInfoOfCurrentDevice();
}

cubble::CudaKernelWrapper::~CudaKernelWrapper()
{}

void cubble::CudaKernelWrapper::generateBubblesOnGPU()
{
    // Get necessary parameters
    int n = env->getNumBubbles();
    int numBlocksPerDim = env->getNumCellsPerDim();
    int totalNumBlocks = numBlocksPerDim * numBlocksPerDim;
    int rngSeed = env->getRngSeed();
    double avgRad = env->getAvgRad();
    double stdDevRad = env->getStdDevRad();
    dvec lbb = env->getLbb();
    dvec tfr = env->getTfr();	

    // Calculate the block & grid sizes based on amount of data & dimensionality
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

    std::cout << "\nAutomatically computed block & grid sizes"
	      << "\nGrid size: (" << gridSize.x
	      << ", " << gridSize.y
	      << ", " << gridSize.z
	      << "), block size: (" << blockSize.x
	      << ", " << blockSize.y
	      << ", " << blockSize.z
	      << ")\nMinimum (total) number of cells with given number of bubbles: "
	      << minNumCells
	      << "\n"
	      << std::endl;
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
    CudaContainer<float> r(n);
    
    CudaContainer<Bubble> b(n);
    CudaContainer<Bubble> reorganizedBubbles(n);
    
    CudaContainer<int> bubblesPerCell(numCells);
    CudaContainer<int> offsets(numCells);

    // Generate random positions & radii
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    CURAND_CALL(curandGenerateUniform(generator, x.getDevicePtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDevicePtr(), n));
    CURAND_CALL(curandGenerateNormal(generator, r.getDevicePtr(), n, avgRad, stdDevRad));
    
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
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
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    bubblesPerCell.toHost();
    bubbleManager->cellEnds.resize(numCells);
    bubbleManager->cellBegins.resize(numCells);

    // Bubbles are stored in a 1D array, even if the domain is 2D or 3D.
    // The offsets to the memory location where the bubbles
    // of the current cell start (and end) are calculated.
    int offset = 0;
    int sharedMemSize = 0;
    for (size_t i = 0; i < numCells; ++i)
    {
	int bubblesInCell = bubblesPerCell[i];
        sharedMemSize = sharedMemSize < bubblesInCell
					? bubblesInCell
					: sharedMemSize;
	    
	offsets[i] = offset;
	bubbleManager->cellBegins[i] = offset;
	offset += bubblesInCell;
	bubbleManager->cellEnds[i] = offset;
    }
    offsets.toDevice();

    // Reset bubblesPerCell to 0, and reuse it.
    bubblesPerCell.fillHostWith(0);
    bubblesPerCell.toDevice();

    // Reorder the bubbles so the bubbles of one cell are stored
    // concurrently in memory.
    assignBubblesToCells<<<gridSize,
	blockSize>>>(b.getDevicePtr(),
		     reorganizedBubbles.getDevicePtr(),
		     offsets.getDevicePtr(),
		     bubblesPerCell.getDevicePtr(),
		     lbb,
		     tfr,
		     n);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    sharedMemSize *= sizeof(Bubble);
    assertMemBelowLimit(sharedMemSize);
    
    findIntersections<<<gridSize, blockSize, sharedMemSize>>>(
	offsets.getDevicePtr(),
	reorganizedBubbles.getDevicePtr(),
	n);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<Bubble> temp;
    reorganizedBubbles.copyDeviceDataToVec(temp);
    bubbleManager->setBubbles(temp);
}

__forceinline__ __device__
int cubble::getGlobalTid()
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

__forceinline__ __device__
int cubble::getBubbleBlockIndex(dvec pos)
{
    int gridX = (int)(pos.x * gridDim.x);
    int gridY = (int)(pos.y * gridDim.y);
    int gridZ = (int)(pos.z * gridDim.z);
    
    return gridZ * gridDim.y * gridDim.x + gridY * gridDim.x + gridX;
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
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	dvec pos;
        pos.x = (double)x[tid];
	pos.y = (double)y[tid];
#if (NUM_DIM == 3)
	pos.z = (double)z[tid];
#endif
	int bbi = getBubbleBlockIndex(pos);
	
	// Scale position
	pos = pos * tfr + lbb;
	
	b[tid].setPos(pos);
	b[tid].setRadius((double)r[tid]);
	
        atomicAdd(&bubblesPerCell[bbi], 1);
    }
}

__global__
void cubble::assignBubblesToCells(Bubble *b,
				  Bubble *reorganizedBubbles,
				  int *offsets,
				  int *currentIndices,
				  dvec lbb,
				  dvec tfr,
				  int numBubbles)
{
    int tid = getGlobalTid();
    dvec interval = tfr - lbb;
    dvec normPos = (b[tid].getPos() - lbb) / interval;

    if (tid < numBubbles)
    {
	int bbi = getBubbleBlockIndex(normPos);
	int offset = atomicAdd(&currentIndices[bbi], 1) + offsets[bbi];
        reorganizedBubbles[offset] = b[tid];

	// Color is used only for debug plot purposes.
	// Can be removed later.
	fvec color((int)(normPos.x * gridDim.x) / (float)gridDim.x,
		   (int)(normPos.y * gridDim.y) / (float)gridDim.y,
		   (int)(normPos.z * gridDim.z) / (float)gridDim.z);
	reorganizedBubbles[offset].setColor(color);
    }
}

__global__
void cubble::findIntersections(int *offsets, Bubble *bubbles, int numBubbles)
{
    extern __shared__ Bubble localBubbles[];
    __shared__ int numOverlapsPerBlock[1];
    
    int numThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x
	+ threadIdx.y * blockDim.x
	+ threadIdx.z * blockDim.x * blockDim.y;

    int numCells = gridDim.x * gridDim.y * gridDim.z;
    int cid = blockIdx.x
	+ blockIdx.y * gridDim.x
	+ blockIdx.z * gridDim.x * gridDim.y;
    
    int numBubblesInCell = cid < numCells - 1
				 ? offsets[cid + 1] - offsets[cid]
				 : numBubbles - offsets[cid];
    int numRounds = numBubblesInCell / numThreads;
    
    for (int round = 0; round <= numRounds; ++round)
    {
	int i = tid + round * numThreads;
	int bid = offsets[cid] + i;
	
	if (i < numBubblesInCell && bid < numBubbles)
	    localBubbles[i] = bubbles[bid];
    }

    if (tid == 0)
	*numOverlapsPerBlock = 0;
    
    __syncthreads();

    for (int round = 0; round <= numRounds; ++round)
    {
	int i = tid + round * numThreads;
	if (i < numBubblesInCell)
	{
	    Bubble *b = &localBubbles[i];
	    double radius1 = b->getRadius();
	    dvec position1 = b->getPos();
	    int numOverlaps = 0;
	    for (int j = 0; j < numBubblesInCell; ++j)
	    {
		if (j == i)
		    continue;
		
		Bubble *b2 = &localBubbles[j];
		double radius2 = b2->getRadius();
		dvec position2 = b2->getPos();

		if ((radius1 + radius2) * (radius1 + radius2)
		    <= (position1 - position2).getSquaredLength())
		    ++numOverlaps;
	    }

	    atomicAdd(&numOverlapsPerBlock[0], numOverlaps);
	}
    }

    printf("Overlaps per block: %d\n", numOverlapsPerBlock[0]);
}
