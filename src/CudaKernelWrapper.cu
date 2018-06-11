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
    
    CudaContainer<Cell> cells(bubbleManager->getCellsSize());
    bubbleManager->getCells(cells);
    
    std::vector<int> indexList;
    std::vector<int> indexListOffsets;
    std::vector<int> indexListSizes;
    indexListOffsets.resize(cells.size());
    indexListSizes.resize(cells.size());

    int offset = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
	std::vector<int> temp;
	bubbleManager->getIndicesFromNeighborCells(temp, i);
	indexList.insert(indexList.end(), temp.begin(), temp.end());
	indexListOffsets[i] = offset;
	indexListSizes[i] = temp.size();
	offset += temp.size();
    }

    int numDomains = 4 + bubbleManager->getNumNeighbors() * 8;
    dim3 gridSize = getGridSize(bubbles.size());
    int gridZ = gridSize.z;
    gridSize.z *= numDomains;

    assertGridSizeBelowLimit(gridSize);

    
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
