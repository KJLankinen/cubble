// -*- C++ -*-

#include "Simulator.h"
#include "Macros.h"
#include "CudaContainer.h"
#include "Cell.h"
#include "Vec.h"

#include <iostream>
#include <sstream>
#include <curand.h>

cubble::Simulator::Simulator(std::shared_ptr<Env> e)
{
    env = e;
    printRelevantInfoOfCurrentDevice();
    
    int n = env->getNumBubbles();
    bubbles = CudaContainer<Bubble>(n);
}

cubble::Simulator::~Simulator()
{}

void cubble::Simulator::setupSimulation()
{
    generateBubbles();
    assignBubblesToCells();
    removeIntersectingBubbles();
    assignBubblesToCells();
    
    bubbles.deviceToHost();
}

double cubble::Simulator::getVolumeOfBubbles() const
{
    double volume = 0;
    for (size_t i = 0; i < bubbles.getSize(); ++i)
    {
	double radius = bubbles[i].getRadius();
#if NUM_DIM == 3
	volume += radius * radius * radius;
#else
	volume += radius * radius;
#endif
    }

    volume *= M_PI;

#if NUM_DIM == 3
    volume *= 1.33333333333333333333333333;
#endif

    return volume;
}

void cubble::Simulator::getBubbles(std::vector<Bubble> &b) const
{
    // Should have some way of knowing which is up to date, device or host memory...
    bubbles.hostToVec(b);
}

void cubble::Simulator::generateBubbles()
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    std::cout << "Starting to generate data for bubbles." << std::flush;
    
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

    // Generate random positions & radii
    std::cout << " Done.\n\tGenerating data..." << std::flush;
    
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

    std::cout << " Done.\n\tAssigning data to bubbles..." << std::flush;;
    
    // Assign generated data to bubbles
    assignDataToBubbles<<<numBlocks, numThreads>>>(x.getDevPtr(),
						   y.getDevPtr(),
#if (NUM_DIM == 3)
						   z.getDevPtr(),
#endif
						   r.getDevPtr(),
						   bubbles.getDevPtr(),
						   lbb,
						   tfr,
						   n);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::cout << " Done." << std::endl;
}

void cubble::Simulator::assignBubblesToCells()
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    std::cout << "Starting to assign bubbles to cells." << std::flush;
    
    int numBubblesPerCell = env->getNumBubblesPerCell();
    dvec tfr = env->getTfr();
    dvec lbb = env->getLbb();
    dim3 gridSize = getGridSize(bubbles.getSize());
    int numCells = gridSize.x * gridSize.y * gridSize.z;
    dvec cellSize = (tfr - lbb) / dvec(gridSize.x, gridSize.y, gridSize.z);
    double minCellSize = 3.0 * env->getAvgRad();

    std::cout << "\n\tUsing grid size (" << gridSize.x
	      << ", " << gridSize.y
	      << ", " << gridSize.z
	      << ") with total of " << numCells << " cells." << std::flush;
    
    if (cellSize.x < minCellSize || cellSize.y < minCellSize || cellSize.z < minCellSize)
    {
	std::stringstream ss;
        ss << "Size of cell (" << cellSize
	   << ") is smaller than the acceptable minimum cell size of "
	   << minCellSize
	   << " in at least one dimension."
	   << "\nEither decrease the number of bubbles or increase the size"
	   << " of the simulation box.";
	
	throw std::runtime_error(ss.str());
    }
    
    cells = CudaContainer<Cell>(numCells);
    indices = CudaContainer<int>(bubbles.getSize());

    std::cout << "\n\tCalculating offsets..." << std::flush;
    
    calculateOffsets<<<gridSize, numBubblesPerCell>>>(bubbles.getDevPtr(),
						      cells.getDevPtr(),
						      lbb,
						      tfr,
						      bubbles.getSize());

    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    cells.deviceToHost();

    int cumulativeSum = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	int numBubbles = cells[i].offset;
        cells[i].offset = cumulativeSum;
	cumulativeSum += numBubbles;
    }
    cells.hostToDevice();

    std::cout << " Done.\n\tAssigning bubbles to cells..." << std::flush;
    
    cubble::assignBubblesToCells<<<gridSize,numBubblesPerCell>>>(bubbles.getDevPtr(),
								 indices.getDevPtr(),
								 cells.getDevPtr(),
								 bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    cells.deviceToHost();

    std::cout << " Done." << std::endl;
}

void cubble::Simulator::removeIntersectingBubbles()
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    std::cout << "Starting the removal of intersecting bubbles." << std::endl;

    dvec tfr = env->getTfr();
    dvec lbb = env->getLbb();
    
    CudaContainer<int> intersections(bubbles.getSize());

    int numThreads = 512;
    int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;
    dim3 gridSize = getGridSize(bubbles.getSize());
    gridSize.z *= numDomains;

    assertGridSizeBelowLimit(gridSize);

    std::cout << "Calculating the size of dynamically allocated shared memory..."
	      << std::flush;

    int sharedMemSize = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	int temp = cells[i].size;
	sharedMemSize = sharedMemSize < temp ? temp : sharedMemSize;
    }
    
    // Nearest even size
    sharedMemSize += sharedMemSize % 2;

    assertMemBelowLimit(sharedMemSize);

    std::cout << " Done.\n\tStarting kernel for finding intersections..." << std::flush;
    
    findIntersections<<<gridSize,
	numThreads,
	sharedMemSize * sizeof(Bubble)>>>(bubbles.getDevPtr(),
					  indices.getDevPtr(),
					  cells.getDevPtr(),
					  intersections.getDevPtr(),
					  tfr,
					  lbb,
					  bubbles.getSize(),
					  numDomains,
					  cells.getSize(),
					  sharedMemSize);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::cout << " Done.\n\tCopying values from device to host..." << std::flush;

    std::vector<Bubble> culledBubbles;
    bubbles.deviceToHost();
    intersections.deviceToHost();
    
    std::cout << " Done.\n\tRemoving intersecting elements from host vector..." << std::flush;

    assert(intersections.getSize() == bubbles.getSize());
    for (size_t i = 0; i < intersections.getSize(); ++i)
    {
	//std::cout << intersections[i] << std::endl;
	if (intersections[i] == 0)
	{
	    culledBubbles.push_back(bubbles[i]);
	}
    }

    std::cout << " Done.\n\tCopying data..." << std::flush;

    bubbles = CudaContainer<Bubble>(culledBubbles);
    
    std::cout << " Done." << std::endl;
}

dim3 cubble::Simulator::getGridSize(int numBubbles)
{
    int numBubblesPerCell = env->getNumBubblesPerCell();
#if NUM_DIM == 3
    int numCellsPerDim = std::ceil(std::cbrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, numCellsPerDim);
#else
    int numCellsPerDim = std::ceil(std::sqrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, 1);
#endif

    return gridSize;
}

__forceinline__ __device__
int cubble::getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum)
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
void cubble::getDomainOffsetsAndIntervals(int numBubbles,
					  int numDomains,
					  int numCells,
					  int numLocalBubbles,
					  ivec cellIdxVec,
					  ivec boxDim,
					  Cell *cells,
					  int &outXBegin,
					  int &outXInterval,
					  int &outYBegin,
					  int &outYInterval)
{
    int domain = blockIdx.z % numDomains;
    int di = (2 * domain) / numDomains;
    
    assert((di == 0 && domain < (int)(0.5f * numDomains))
	   || (di == 1 && domain >= (int)(0.5f * numDomains)));
    
    int dj = domain % (int)(0.5f * numDomains);
    int djMod2 = dj % 2;

    // Find this cell
    int selfCellIndex = cellIdxVec.z * boxDim.x * boxDim.y
	+ cellIdxVec.y * boxDim.x
	+ cellIdxVec.x;
    assert(selfCellIndex < numCells);
    Cell self = cells[selfCellIndex];

    // Find the neighbor of this cell
    int neighborCellIndex = getNeighborCellIndex(cellIdxVec, boxDim, dj / 2);
    assert(neighborCellIndex < numCells);
    Cell neighbor = cells[neighborCellIndex];

    // Find the interval of values to use:
    // x-axis uses the right or the left half of the neighbor cell
    int halfSize = 0.5f * neighbor.size;
    outXBegin = neighbor.offset + djMod2 * halfSize;
    outXInterval = halfSize + djMod2 * (neighbor.size % 2);
    
    assert(outXBegin + outXInterval <= numBubbles);
    assert(outXBegin + outXInterval <= neighbor.size + neighbor.offset);
    assert(outXInterval == halfSize || outXInterval == halfSize + 1);

    // y-axis uses the top or bottom half of this cell
    halfSize = 0.5f * self.size;
    outYBegin = self.offset + di * halfSize;
    outYInterval = halfSize + di * (self.size % 2);

    assert(outYBegin + outYInterval <= numBubbles);
    assert(outYInterval == halfSize || outYInterval == halfSize + 1);
    assert(outYBegin + outYInterval <= self.size + self.offset);
    assert(outXInterval + outYInterval <= numLocalBubbles);
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
double cubble::getWrappedSquaredLength(dvec tfr, dvec lbb, dvec pos1, dvec pos2)
{
    dvec temp = pos1 - pos2;
    double length = temp.getSquaredLength();
    dvec interval = tfr - lbb;

    pos2.x = temp.x < -0.5 * interval.x ? pos2.x - interval.x
		      : (temp.x > 0.5 * interval.x ? pos2.x + interval.x  : pos2.x);
    pos2.y = temp.y < -0.5 * interval.y ? pos2.y - interval.y
		      : (temp.y > 0.5 * interval.y ? pos2.y + interval.y  : pos2.y);
    pos2.z = temp.z < -0.5 * interval.z ? pos2.z - interval.z
		      : (temp.z > 0.5 * interval.z ? pos2.z + interval.z  : pos2.z);

    double length2 = (pos1 - pos2).getSquaredLength();
    assert(length2 <= length);
    
    return length2;
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
        indices[offset] = tid;
    }
}

__global__
void cubble::findIntersections(Bubble *bubbles,
			       int *indices,
			       Cell *cells,
			       int *intersectingIndices,
			       dvec tfr,
			       dvec lbb,
			       int numBubbles,
			       int numDomains,
			       int numCells,
			       int numLocalBubbles)
{
    extern __shared__ Bubble localBubbles[];

    assert(numBubbles > 0);
    assert(numDomains > 0);
    assert(numCells > 0);
    assert(numLocalBubbles > 0);
    assert(!(numDomains & 1));
    
    ivec cellIdxVec(blockIdx.x, blockIdx.y, blockIdx.z / numDomains);
    ivec boxDim(gridDim.x, gridDim.y, gridDim.z / numDomains);

    int xBegin = -1;
    int xInterval = -1;
    int yBegin = -1;
    int yInterval = -1;
    
    getDomainOffsetsAndIntervals(numBubbles,
				 numDomains,
				 numCells,
				 numLocalBubbles,
				 cellIdxVec,
				 boxDim,
				 cells,
				 xBegin,
				 xInterval,
				 yBegin,
				 yInterval);

    assert(xBegin >= 0 && xInterval > 0 && yBegin >= 0 && yInterval > 0);
    
    // Get the bubbles to shared memory
    if (threadIdx.x < xInterval + yInterval)
    {
	if (threadIdx.x < xInterval)
	    localBubbles[threadIdx.x] = bubbles[indices[xBegin + threadIdx.x]];
	else
	    localBubbles[threadIdx.x] = bubbles[indices[yBegin + (threadIdx.x - xInterval)]];
    }

    __syncthreads();
      
    int numPairs = xInterval * yInterval;
    int numRounds = 1 + (numPairs / blockDim.x);

    for (int round = 0; round < numRounds; ++round)
    {
        int pairIdx = round * blockDim.x + threadIdx.x;
	if (pairIdx < numPairs)
	{
	    int x = pairIdx % xInterval;
	    int y = pairIdx / xInterval;
	    assert(y < yInterval);

	    int gid1 = xBegin + x;
	    int gid2 = yBegin + y;
	    
	    // Skip self-intersection
	    if (gid1 == gid2)
		continue;
	    
	    Bubble *b1 = &localBubbles[x];
	    Bubble *b2 = &localBubbles[xInterval + y];
	    
	    double radii = b1->getRadius() + b2->getRadius();
	    double length = getWrappedSquaredLength(tfr, lbb, b1->getPos(), b2->getPos());
	    
	    if (radii * radii > length)
	    {
		gid1 = indices[gid1];
		gid2 = indices[gid2];
		int gid = gid1 < gid2 ? gid1 : gid2;
		
		atomicAdd(&intersectingIndices[gid], 1);
	    }
	}
    }
}
