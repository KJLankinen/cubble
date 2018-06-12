// -*- C++ -*-

#include "CudaKernelWrapper.h"
#include "Macros.h"
#include "CudaContainer.h"
#include "Cell.h"

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

void cubble::CudaKernelWrapper::generateBubbles(std::vector<Bubble> &outBubbles)
{
    std::cout << "Generating bubbles..." << std::endl;
    
    // Get necessary parameters
    int n = env->getNumBubbles();
    int rngSeed = env->getRngSeed();
    double avgRad = env->getAvgRad();
    double stdDevRad = env->getStdDevRad();
    dvec lbb = env->getLbb();
    dvec tfr = env->getTfr();	

    int numThreads = 1024;
    int numBlocks = (int)std::ceil(n / (float)numThreads);
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
    CudaContainer<float> r(n);
    
    CudaContainer<Bubble> b(n);

    // Generate random positions & radii
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    CURAND_CALL(curandGenerateUniform(generator, x.getDevPtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDevPtr(), n));
    CURAND_CALL(curandGenerateNormal(generator, r.getDevPtr(), n, avgRad, stdDevRad));
    
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
    CURAND_CALL(curandGenerateUniform(generator, z.getDevPtr(), n));
#endif
    
    // Assign generated data to bubbles
    assignDataToBubbles<<<numBlocks, numThreads>>>(x.getDevPtr(),
						 y.getDevPtr(),
#if (NUM_DIM == 3)
						 z.getDevPtr(),
#endif
						 r.getDevPtr(),
						 b.getDevPtr(),
						 lbb,
						 tfr,
						 n);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    b.copyDeviceDataToVec(outBubbles);
}

void cubble::CudaKernelWrapper::assignBubblesToCells(const std::vector<Bubble> &b)
{
    std::cout << "Assigning bubbles to cells..." << std::endl;
    
    int numBubblesPerCell = env->getNumBubblesPerCell();
    dvec tfr = env->getTfr();
    dvec lbb = env->getLbb();
    dim3 gridSize = getGridSize(b.size());
    int numCells = gridSize.x * gridSize.y * gridSize.z;

    std::cout << "Grid size: (" << gridSize.x
	      << ", " << gridSize.y
	      << ", " << gridSize.z
	      << "), numCells: " << numCells
	      << std::endl;
    
    CudaContainer<Bubble> bubbles(b);
    CudaContainer<Cell> cells(numCells);
    CudaContainer<int> indices(bubbles.size());

    bubbles.toDevice();
    cells.toDevice();
    
    calculateOffsets<<<gridSize, numBubblesPerCell>>>(bubbles.getDevPtr(),
						      cells.getDevPtr(),
						      lbb,
						      tfr,
						      bubbles.size());

    cells.toHost();

    int cumulativeSum = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
	int numBubbles = cells[i].offset;
        cells[i].offset = cumulativeSum;
	cumulativeSum += numBubbles;
    }
    cells.toDevice();
    
    cubble::assignBubblesToCells<<<gridSize,numBubblesPerCell>>>(bubbles.getDevPtr(),
								 indices.getDevPtr(),
								 cells.getDevPtr(),
								 bubbles.size());
    
    bubbleManager->setBubblesFromDevice(bubbles);
    bubbleManager->setIndicesFromDevice(indices);
    bubbleManager->setCellsFromDevice(cells);
}

void cubble::CudaKernelWrapper::removeIntersectingBubbles(const std::vector<Bubble> &b)
{
    std::cout << "Removing intersecing bubbles..." << std::endl;
    
    CudaContainer<Bubble> bubbles(b);
    CudaContainer<int> indices(bubbles.size());
    CudaContainer<int> intersections(bubbles.size() + 1); // <-- First value gives count
    CudaContainer<Cell> cells(bubbleManager->getCellsSize());
    bubbleManager->getCells(cells);
    bubbleManager->getIndices(indices);
    intersections.fillHostWith(0);

    int numThreads = 1024;
    int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;
    dim3 gridSize = getGridSize(bubbles.size());
    int originalGridZ = gridSize.z;
    gridSize.z *= numDomains;

    assertGridSizeBelowLimit(gridSize);

    bubbles.toDevice();
    indices.toDevice();
    intersections.toDevice();
    cells.toDevice();

    int sharedMemSize = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
	int temp = cells[i].size;
	sharedMemSize = sharedMemSize < temp ? temp : sharedMemSize;
    }
    sharedMemSize = (int)(sharedMemSize * 0.5f) + 1;
    sharedMemSize *= sizeof(Bubble);

    assertMemBelowLimit(sharedMemSize);
    
    findIntersections<<<numThreads, gridSize, sharedMemSize>>>(bubbles.getDevPtr(),
							       indices.getDevPtr(),
							       cells.getDevPtr(),
							       intersections.getDevPtr(),
							       bubbles.size(),
							       numDomains);
}

dim3 cubble::CudaKernelWrapper::getGridSize(int numBubbles)
{
    int numBubblesPerCell = env->getNumBubblesPerCell();
#if NUM_DIM == 3
    int numCellsPerDim = (int)std::ceil(std::cbrt(numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, numCellsPerDim);
#else
    int numCellsPerDim = (int)std::ceil(std::sqrt(numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, 1);
#endif

    return gridSize;
}

__forceinline__ __device__
int cubble::geNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum)
{
    // Switch statements and ifs that diverge inside one warp/block are
    // detrimental for performance. However, this should never diverge,
    // as all the threads of one block should always be in the same cell
    // going for the same neighbor.
    ivec idxVec = cellIdx;
    switch(neighborNum)
    {
    case 0:
	// self
	break;
    case 1:
	idxVec += ivec(-1, 1, 0);
	break;
    case 2:
	idxVec += ivec(-1, 0, 0);
	break;
    case 3:
	idxVec += ivec(-1, -1, 0);
	break;
    case 4:
	idxVec += ivec(0, -1, 0);
	break;
#if NUM_DIM == 3
    case 5:
	idxVec += ivec(-1, 1, -1);
	break;
    case 6:
	idxVec += ivec(-1, 0, -1);
	break;
    case 7:
	idxVec += ivec(-1, -1, -1);
	break;
    case 8:
	idxVec += ivec(0, 1, -1);
	break;
    case 9:
	idxVec += ivec(0, 0, -1);
	break;
    case 10:
	idxVec += ivec(0, -1, -1);
	break;
    case 11:
	idxVec += ivec(1, 1, -1);
	break;
    case 12:
	idxVec += ivec(1, 0, -1);
	break;
    case 13:
	idxVec += ivec(1, -1, -1);
	break;
#endif
    default:
	printf("Should never end up here!");
	break;
    }

    idxVec += dim;
    idxVec %= dim;

    return idxVec.z * dim.y * dim.x + idxVec.y * dim.x + idxVec.x;
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

__global__
void cubble::assignDataToBubbles(float *x,
				 float *y,
#if (NUM_DIM == 3)
				 float *z,
#endif
				 float *r,
				 Bubble *b,
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
	// Scale position
	pos = pos * tfr + lbb;
	
	b[tid].setPos(pos);
	b[tid].setRadius((double)r[tid]);
    }
}

__global__
void cubble::calculateOffsets(Bubble *bubbles,
			      Cell *cells,
			      dvec lbb,
			      dvec tfr,
			      int numBubbles)
{   
    int tid = getGlobalTid();
    dvec invInterval = 1.0 / (tfr - lbb);
    
    if (tid < numBubbles)
    {
	dvec pos = (bubbles[tid].getPos() - lbb) * invInterval;
	ivec indexVec(gridDim.x * pos.x, gridDim.y * pos.y, gridDim.z * pos.z);
	int index = gridDim.x * gridDim.y * indexVec.z
	    + gridDim.x * indexVec.y
	    + indexVec.x;

	fvec color = fvec(indexVec.x / (float)gridDim.x,
			  indexVec.y / (float)gridDim.y,
			  indexVec.z / (float)gridDim.z);
	bubbles[tid].setCellIndex(index);
        bubbles[tid].setColor(color);
	
	atomicAdd(&cells[index].offset, 1);
    }
}


__global__
void cubble::assignBubblesToCells(Bubble *bubbles,
				  int *indices,
				  Cell *cells,
				  int numBubbles)
{
    int tid = getGlobalTid();

    if (tid < numBubbles)
    {
	int index = bubbles[tid].getCellIndex();
	int offset = cells[index].offset + atomicAdd(&cells[index].size, 1);
        indices[offset] = bubbles[tid].getCellIndex();
    }
}

__global__
void cubble::findIntersections(Bubble *bubbles,
			       int *indices,
			       Cell *cells,
			       int *intersectingIndices,
			       int numBubbles,
			       int numDomains)
{
    extern __shared__ Bubble localBubbles[];
    ivec cellIdx(blockIdx.x, blockIdx.y, blockIdx.z / numDomains);
    ivec boxDim(gridDim.x, gridDim.y, gridDim.z / numDomains);
    int domain = blockIdx.z % numDomains;
    int halfNumDomains = numDomains / 2;
    int di = domain / halfNumDomains;
    int dj = domain % halfNumDomains;
    int djMod2 = dj % 2;

    __shared__ Cell neighbor;
    __shared__ Cell self;
    if (threadIdx.x == 0)
    {
	int temp = cellIdx.z * boxDim.x * boxDim.y + cellIdx.y * boxDim.x + cellIdx.x;
	self = cells[temp];
	neighbor = cells[getNeighborCell(cellIdx, boxDim, dj / 2)];
    }
    __syncthreads();

    int xBegin = neighbor.offset;
    xBegin += djMod2 * neighbor.size * 0.5f;
    int xEnd = xBegin + neighbor.size * (0.5f + 0.5f * djMod2);
    int xInterval = xEnd - xBegin;

    int yBegin = self.offset;
    yBegin += di * self.size * 0.5f;
    int yEnd = YBegin + self.size * (0.5f + 0.5f * di);
    int yInterval = yEnd - yBegin;

    if (threadIdx.x < xInterval && xBegin + threadIdx.x < numBubbles)
	localBubbles[threadIdx.x] = bubbles[indices[xBegin + threadIdx.x]];
    else if (threadIdx.x < xInterval + yInterval && yBegin + threadIdx.x < numBubbles)
	localBubbles[threadIdx.x] = bubbles[indices[yBegin + threadIdx.x]];

    __syncthreads();

    int numPairs = xInterval * yInterval;
    int numRounds = numPairs / blockDim.x + 1;

    for (int round = 0; round < numRounds; ++round)
    {
        int tempIdx = round * blockDim.x + threadIdx.x;
	if (tempIdx < numPairs)
	{
	    int x = tempIdx % xInterval;
	    int y = tempIdx / xInterval;

	    if (x > xInterval)
		printf("", );
	    
	    if (y > yInterval)
		printf("", );
	    
	    Bubble *b1 = &localBubbles[x];
	    Bubble *b2 = &localBubbles[xInterval + y];

	    // Skip self-intersections
	    if (b1 == b2)
	    {
		THIS DOES NOT DETECT SAME BUBBLE CORRECTLY, SINCE THERE ARE DUPLICATES.
		continue;
	    }
	    
	    double radii = b1->getRadius() + b2->getRadius();
	    if (radii * radii < (b1->getPos() - b2->getPos).getSquaredLength())
		intersectingIndices[indices[xBegin + x]] = 1;
	}
    }
}
