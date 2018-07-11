// -*- C++ -*-

#include "Simulator.h"
#include "Macros.h"
#include "CudaContainer.h"
#include "Cell.h"
#include "Vec.h"
#include "Util.h"

#include <iostream>
#include <sstream>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


// ******************************
// Class functions run on CPU
// ******************************


cubble::Simulator::Simulator(std::shared_ptr<Env> e)
{
    env = e;
    printRelevantInfoOfCurrentDevice();
}

cubble::Simulator::~Simulator()
{}

void cubble::Simulator::setupSimulation()
{
    generateBubbles();
    assignBubblesToCells(true);
}

void cubble::Simulator::integrate(bool useGasExchange)
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    dim3 gridSize = getGridSize(bubbles.getSize());
    double timeStep = env->getTimeStep();
    double error = 0;
    
    int maxNumBubbles = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	int temp = cells[i].size;
	maxNumBubbles = maxNumBubbles < temp ? temp : maxNumBubbles;
    }

    assert(maxNumBubbles > 0);
    // Nearest even size
    maxNumBubbles += maxNumBubbles % 2;
    assertMemBelowLimit(maxNumBubbles * sizeof(Bubble));
    int numThreads = maxNumBubbles;

    do
    {
        // Calculate prediction
	predict<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    tfr,
	    lbb,
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize());
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	// Calculate accelerations
	accelerations = CudaContainer<dvec>(bubbles.getSize());
	
	accelerate<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    numberOfNeighbors.getDataPtr(),
	    neighborIndices.getDataPtr(),
	    accelerations.getDataPtr(),
	    energies.getDataPtr(),
	    tfr,
	    lbb,
	    bubbles.getSize(),
	    cells.getSize(),
	    neighborStride);
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	correct<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    errors.getDataPtr(),
	    accelerations.getDataPtr(),
	    tfr,
	    lbb,
	    env->getFZeroPerMuZero(),
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize());
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
        
        error = *thrust::max_element(thrust::host,
					    errors.getDataPtr(),
					    errors.getDataPtr() + errors.getSize());
	
	if (error < env->getErrorTolerance() / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > env->getErrorTolerance())
	    timeStep *= 0.5;
    }
    while (error > env->getErrorTolerance());

    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    ElasticEnergy = thrust::reduce(thrust::host,
				   energies.getDataPtr(),
				   energies.getDataPtr() + energies.getSize());

    updateData<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	bubbles.getDataPtr(),
	indices.getDataPtr(),
	cells.getDataPtr(),
        bubbles.getSize(),
        cells.getSize());
	
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

double cubble::Simulator::getVolumeOfBubbles() const
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    double volume = 0;
    CudaContainer<double> volumes(bubbles.getSize());
    int numThreads = 1024;
    int numBlocks = (int)std::ceil(bubbles.getSize() / (float)numThreads);

    calculateVolumes<<<numBlocks, numThreads>>>(bubbles.getDataPtr(),
						volumes.getDataPtr(),
						bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    volume = thrust::reduce(thrust::host,
			    volumes.getDataPtr(),
			    volumes.getDataPtr() + volumes.getSize());
    
    return volume;
}

void cubble::Simulator::getBubbles(std::vector<Bubble> &b) const
{ 
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    bubbles.dataToVec(b);
}

void cubble::Simulator::generateBubbles()
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    std::cout << "Starting to generate data for bubbles." << std::endl;
    
    const int rngSeed = env->getRngSeed();
    const double avgRad = env->getAvgRad();
    const double stdDevRad = env->getStdDevRad();
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    ivec bubblesPerDim = 0.45 * (tfr - lbb) / avgRad;
    int n = bubblesPerDim.x * bubblesPerDim.y;
    
    dim3 blocksPerGrid(1, 1, 1);
#if (NUM_DIM == 3)
    n *= bubblesPerDim.z;
    dim3 threadsPerBlock(10, 10, 10);
    blocksPerGrid.z = (int)std::ceil(bubblesPerDim.z / (float)threadsPerBlock.z);
#else
    dim3 threadsPerBlock(32, 32, 1);
#endif
    
    bubbles = CudaContainer<Bubble>(n);

    blocksPerGrid.x = (int)std::ceil(bubblesPerDim.x / (float)threadsPerBlock.x);
    blocksPerGrid.y = (int)std::ceil(bubblesPerDim.y / (float)threadsPerBlock.y);
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
    CudaContainer<float> w(n);
    CudaContainer<float> r(n);

    std::cout << "\tGenerating data..." << std::endl;
    
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    CURAND_CALL(curandGenerateUniform(generator, x.getDataPtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDataPtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, w.getDataPtr(), n));
    CURAND_CALL(curandGenerateNormal(generator, r.getDataPtr(), n, avgRad, stdDevRad));
    
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
    CURAND_CALL(curandGenerateUniform(generator, z.getDataPtr(), n));
#endif
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::cout << "\tAssigning data to bubbles..." << std::endl;;
    
    assignDataToBubbles<<<blocksPerGrid, threadsPerBlock>>>(x.getDataPtr(),
							    y.getDataPtr(),
#if (NUM_DIM == 3)
							    z.getDataPtr(),
#endif
							    r.getDataPtr(),
							    w.getDataPtr(),
							    bubbles.getDataPtr(),
							    bubblesPerDim,
							    tfr,
							    lbb,
							    avgRad,
							    n);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

void cubble::Simulator::assignBubblesToCells(bool useVerboseOutput)
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    if (useVerboseOutput)
	std::cout << "Starting to assign bubbles to cells." << std::endl;
    
    const int numBubblesPerCell = env->getNumBubblesPerCell();
    dim3 gridSize = getGridSize(bubbles.getSize());
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec cellSize = (env->getTfr() - env->getLbb()) /
	dvec(gridSize.x, gridSize.y, gridSize.z);
    const double minCellSize = 3.0 * env->getAvgRad();
    
    if (useVerboseOutput)
	std::cout << "\tUsing grid size (" << gridSize.x
		  << ", " << gridSize.y
		  << ", " << gridSize.z
		  << ") with total of " << numCells << " cells." << std::endl;

#if NUM_DIM == 3    
    if (cellSize.x < minCellSize || cellSize.y < minCellSize || cellSize.z < minCellSize)
#else
    if (cellSize.x < minCellSize || cellSize.y < minCellSize)
#endif
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
    errors = CudaContainer<double>(bubbles.getSize());
    energies = CudaContainer<double>(bubbles.getSize());
    numberOfNeighbors = CudaContainer<int>(bubbles.getSize());
    neighborIndices = CudaContainer<int>(bubbles.getSize() * neighborStride);

    if (useVerboseOutput)
	std::cout << "\tCalculating offsets..." << std::endl;
    
    calculateOffsets<<<gridSize, numBubblesPerCell>>>(bubbles.getDataPtr(),
						      cells.getDataPtr(),
						      bubbles.getSize());

    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    int maxNumBubbles = 0;
    int cumulativeSum = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	int numBubbles = cells[i].offset;
        cells[i].offset = cumulativeSum;
	cumulativeSum += numBubbles;
	maxNumBubbles = numBubbles > maxNumBubbles ? numBubbles : maxNumBubbles;
    }
    
    maxNumBubbles += maxNumBubbles % 2;
    assertMemBelowLimit(maxNumBubbles * sizeof(Bubble));
    
    if (useVerboseOutput)
	std::cout << "\tAssigning bubbles to cells..." << std::endl;
    
    bubblesToCells<<<gridSize,numBubblesPerCell>>>(bubbles.getDataPtr(),
						   indices.getDataPtr(),
						   cells.getDataPtr(),
						   bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    int numThreads = 512;
    int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;
    gridSize = getGridSize(bubbles.getSize());
    gridSize.z *= numDomains;
    assertGridSizeBelowLimit(gridSize);

    if (useVerboseOutput)
	std::cout << "\tFinding neighbors for each bubble..." << std::endl;
    
    findNeighbors<<<gridSize, numThreads, maxNumBubbles * sizeof(Bubble)>>>(
	bubbles.getDataPtr(),
	indices.getDataPtr(),
	cells.getDataPtr(),
        numberOfNeighbors.getDataPtr(),
	neighborIndices.getDataPtr(),
	env->getTfr(),
	env->getLbb(),
	bubbles.getSize(),
	numDomains,
	cells.getSize(),
	maxNumBubbles,
	neighborStride);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
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


// ******************************
// Kernels
// ******************************


__global__
void cubble::calculateVolumes(Bubble *b, double *volumes, int numBubbles)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double radius = b[tid].getRadius();
	double volume = radius * radius * 3.14159265359;
#if (NUM_DIM == 3)
	volume *= radius * 1.33333333333333333333333333;
#endif
	
	volumes[tid] = volume;
    }   
}

__global__
void cubble::assignDataToBubbles(float *x,
				 float *y,
#if (NUM_DIM == 3)
				 float *z,
#endif
				 float *r,
				 float *w,
				 Bubble *b,
				 ivec bubblesPerDim,
				 dvec tfr,
				 dvec lbb,
				 double avgRad,
				 int numBubbles)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    int zid = blockIdx.z * blockDim.z + threadIdx.z;

    int gid = zid * bubblesPerDim.x * bubblesPerDim.y + yid * bubblesPerDim.x + xid;

    if (gid < numBubbles
	&& xid < bubblesPerDim.x
	&& yid < bubblesPerDim.y
	&& zid < bubblesPerDim.z)
    {
	dvec randomOffset(x[gid], y[gid], 0);
	dvec pos(0, 0, 0);
	pos.x = xid / (float)bubblesPerDim.x;
	pos.y = yid / (float)bubblesPerDim.y;
#if (NUM_DIM == 3)
	pos.z = zid / (float)bubblesPerDim.z;
	randomOffset.z = z[gid];
#endif

	randomOffset = dvec::normalize(randomOffset) * w[gid] * avgRad;
	randomOffset = (randomOffset - lbb) / (tfr - lbb);
	pos = getWrappedPos(pos + randomOffset);

	b[gid].setPos(pos);
	b[gid].setRadius(r[gid]);
    }
}

__global__
void cubble::calculateOffsets(Bubble *bubbles,
			      Cell *cells,
			      int numBubbles)
{   
    int tid = getGlobalTid();
    
    if (tid < numBubbles)
    {
	dvec pos = bubbles[tid].getPos();
	ivec indexVec(gridDim.x * pos.x, gridDim.y * pos.y, gridDim.z * pos.z);
	int index = gridDim.x * gridDim.y * indexVec.z
	    + gridDim.x * indexVec.y
	    + indexVec.x;

	bubbles[tid].setCellIndex(index);
	atomicAdd(&cells[index].offset, 1);
    }
}

__global__
void cubble::bubblesToCells(Bubble *bubbles,
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
void cubble::findNeighbors(Bubble *bubbles,
			   int *indices,
			   Cell *cells,
			   int *numberOfNeighbors,
			   int *neighborIndices,
			   dvec tfr,
			   dvec lbb,
			   int numBubbles,
			   int numDomains,
			   int numCells,
			   int numLocalBubbles,
			   int neighborStride)
{
    extern __shared__ Bubble localBubbles[];
    
    DEVICE_ASSERT(numBubbles > 0);
    DEVICE_ASSERT(numDomains > 0);
    DEVICE_ASSERT(numCells > 0);
    DEVICE_ASSERT(numLocalBubbles > 0);
    DEVICE_ASSERT(!(numDomains & 1));
    
    ivec cellIdxVec(blockIdx.x, blockIdx.y, blockIdx.z / numDomains);
    ivec boxDim(gridDim.x, gridDim.y, gridDim.z / numDomains);
    
    int xBegin = -1;
    int xInterval = -1;
    int yBegin = -1;
    int yInterval = -1;
    bool isOwnCell = false;
    
    getDomainOffsetsAndIntervals(numBubbles,
				 numDomains,
				 numCells,
				 cellIdxVec,
				 boxDim,
				 cells,
				 xBegin,
				 xInterval,
				 yBegin,
				 yInterval,
				 isOwnCell);
    
    DEVICE_ASSERT(xBegin >= 0 && xInterval > 0 && yBegin >= 0 && yInterval > 0);
    DEVICE_ASSERT(xInterval + yInterval <= numLocalBubbles);
    
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
	    DEVICE_ASSERT(y < yInterval);
	    
	    int gid1 = indices[xBegin + x];
	    int gid2 = indices[yBegin + y];
	    
	    if (gid1 == gid2)
		continue;
	    
	    const Bubble *b1 = &localBubbles[x];
	    const Bubble *b2 = &localBubbles[xInterval + y];
	    
	    double radii = b1->getRadius() + b2->getRadius();
	    dvec posVec = getShortestWrappedNormalizedVec(b1->getPos(), b2->getPos());
	    double length = (posVec * (tfr - lbb)).getSquaredLength();
	    
	    if (radii * radii > length)
	    {
		int index = atomicAdd(&numberOfNeighbors[gid1], 1);
		DEVICE_ASSERT(index < neighborStride);
		index += neighborStride * gid1;
		DEVICE_ASSERT(index < numBubbles * neighborStride);
		neighborIndices[index] = gid2;

		if (!isOwnCell)
		{
		    index = atomicAdd(&numberOfNeighbors[gid2], 1);
		    DEVICE_ASSERT(index < neighborStride);
		    index += neighborStride * gid2;
		    DEVICE_ASSERT(index < numBubbles * neighborStride);
		    neighborIndices[index] = gid1;
		}
	    }
	}
    }
}

__global__
void cubble::predict(Bubble *bubbles,
		     int *indices,
		     Cell *cells,
		     dvec tfr,
		     dvec lbb,
		     double timeStep,
		     int numBubbles,
		     int numCells)
{
    extern __shared__ Bubble localBubbles[];

    int cid = blockIdx.z * gridDim.x * gridDim.y
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;

    DEVICE_ASSERT(cid < numCells);
    const Cell *self = &cells[cid];

    if (threadIdx.x < self->size)
    {
	localBubbles[threadIdx.x] = bubbles[indices[self->offset + threadIdx.x]];
        Bubble *bubble = &localBubbles[threadIdx.x];

	// Scale, predict, normalize, enfore boundaries
	dvec interval = (tfr - lbb);
	dvec pos = lbb + bubble->getPos() * interval;
	pos += 0.5 * timeStep * (3.0 * bubble->getVel() - bubble->getVelPrev());
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);
	bubble->setPosPred(pos);
	
	bubbles[indices[self->offset + threadIdx.x]] = localBubbles[threadIdx.x];
    }
}

__global__
void cubble::correct(Bubble *bubbles,
		     int *indices,
		     Cell *cells,
		     double *errors,
		     dvec *accelerations,
		     dvec tfr,
		     dvec lbb,
		     double fZeroPerMuZero,
		     double timeStep,
		     int numBubbles,
		     int numCells)
{
    extern __shared__ Bubble localBubbles[];

    int cid = blockIdx.z * gridDim.x * gridDim.y
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;

    DEVICE_ASSERT(cid < numCells);
    const Cell *self = &cells[cid];

    if (threadIdx.x < self->size)
    {
	int gid = indices[self->offset + threadIdx.x];
	localBubbles[threadIdx.x] = bubbles[gid];
        Bubble *bubble = &localBubbles[threadIdx.x];

	bubble->setVelPred(accelerations[gid] * fZeroPerMuZero);
	
	// Scale, correct, normalize, enforce boundaries
	dvec interval = (tfr - lbb);
	dvec pos = lbb + bubble->getPos() * interval;
	pos += 0.5 * timeStep * (bubble->getVel() + bubble->getVelPred());
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);

	double error = (pos - bubble->getPosPred()).getAbsolute().getMaxComponent();
	errors[gid] = error;
	bubble->setPosPred(pos);
	bubbles[gid] = localBubbles[threadIdx.x];
    }
}

__global__
void cubble::accelerate(Bubble *bubbles,
			int *indices,
			Cell *cells,
			int *numberOfNeighbors,
			int *neighborIndices,
		        dvec *accelerations,
			double *energies,
			dvec tfr,
			dvec lbb,
			int numBubbles,
			int numCells,
			int neighborStride)
{
    extern __shared__ Bubble localBubbles[];

    const int cid = blockIdx.z * gridDim.x * gridDim.y
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;

    DEVICE_ASSERT(cid < numCells);
    const Cell *self = &cells[cid];

    if (threadIdx.x < self->size)
    {
	const int gid = indices[self->offset + threadIdx.x];
	localBubbles[threadIdx.x] = bubbles[gid];
        const Bubble *bubble = &localBubbles[threadIdx.x];

	dvec acceleration(0, 0, 0);
	double energy = 0;
	
	for (int i = 0; i < numberOfNeighbors[gid]; ++i)
	{
	    const int index = neighborIndices[gid * neighborStride + i];
	    DEVICE_ASSERT(index < numBubbles);
	    const Bubble *neighbor = &bubbles[index];

	    const double radii = bubble->getRadius() + neighbor->getRadius();
            dvec distance = getShortestWrappedNormalizedVec(bubble->getPosPred(),
							    neighbor->getPosPred());
            distance *= (tfr - lbb);
	    const double magnitude = distance.getLength();
	    const double temp = radii - magnitude;
	    distance *= temp / (radii * magnitude);
	    energy += (temp * temp) / radii;
	    acceleration += distance;
	}

	accelerations[gid] = acceleration;
	energies[gid] = energy;
    }
}

__global__
void cubble::updateData(Bubble *bubbles,
			int *indices,
			Cell *cells,
			int numBubbles,
			int numCells)
{
    extern __shared__ Bubble localBubbles[];

    int cid = blockIdx.z * gridDim.x * gridDim.y
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;

    DEVICE_ASSERT(cid < numCells);
    const Cell *self = &cells[cid];

    if (threadIdx.x < self->size)
    {
	int gid = indices[self->offset + threadIdx.x];
	localBubbles[threadIdx.x] = bubbles[gid];
        Bubble *bubble = &localBubbles[threadIdx.x];

	bubble->setPosPrev(bubble->getPos());
	bubble->setPos(bubble->getPosPred());
	bubble->setVelPrev(bubble->getVel());
	bubble->setVel(bubble->getVelPred());
	
	bubbles[gid] = localBubbles[threadIdx.x];
    }
}


// ******************************
// Device functions
// ******************************


__device__
double cubble::atomicAddD(double *address, double val)
{
    unsigned long long int *addressAsUll = (unsigned long long int*)address;
    unsigned long long int old = *addressAsUll;
    unsigned long long int assumed = old;
    
    do
    {
	assumed = old;
	old = atomicCAS(addressAsUll,
			assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    
    return __longlong_as_double(old);
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
					  ivec cellIdxVec,
					  ivec boxDim,
					  Cell *cells,
					  int &outXBegin,
					  int &outXInterval,
					  int &outYBegin,
					  int &outYInterval,
					  bool &outIsOwnCell)
{
    int domain = blockIdx.z % numDomains;
    int di = (2 * domain) / numDomains;
    
    DEVICE_ASSERT((di == 0 && domain < (int)(0.5f * numDomains))
	   || (di == 1 && domain >= (int)(0.5f * numDomains)));
    
    int dj = domain % (int)(0.5f * numDomains);
    int djMod2 = dj % 2;

    // Find this cell
    int selfCellIndex = cellIdxVec.z * boxDim.x * boxDim.y
	+ cellIdxVec.y * boxDim.x
	+ cellIdxVec.x;
    DEVICE_ASSERT(selfCellIndex < numCells);
    Cell self = cells[selfCellIndex];

    // Find the neighbor of this cell
    int neighborCellIndex = getNeighborCellIndex(cellIdxVec, boxDim, dj / 2);
    DEVICE_ASSERT(neighborCellIndex < numCells);
    Cell neighbor = cells[neighborCellIndex];
    
    outIsOwnCell = selfCellIndex == neighborCellIndex;

    // Find the interval of values to use:
    // x-axis uses the right or the left half of the neighbor cell
    int halfSize = 0.5f * neighbor.size;
    outXBegin = neighbor.offset + djMod2 * halfSize;
    outXInterval = halfSize + djMod2 * (neighbor.size % 2);
    
    DEVICE_ASSERT(outXBegin + outXInterval <= numBubbles);
    DEVICE_ASSERT(outXBegin + outXInterval <= neighbor.size + neighbor.offset);
    DEVICE_ASSERT(outXInterval == halfSize || outXInterval == halfSize + 1);

    // y-axis uses the top or bottom half of this cell
    halfSize = 0.5f * self.size;
    outYBegin = self.offset + di * halfSize;
    outYInterval = halfSize + di * (self.size % 2);

    DEVICE_ASSERT(outYBegin + outYInterval <= numBubbles);
    DEVICE_ASSERT(outYInterval == halfSize || outYInterval == halfSize + 1);
    DEVICE_ASSERT(outYBegin + outYInterval <= self.size + self.offset);
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
cubble::dvec cubble::getShortestWrappedNormalizedVec(dvec pos1, dvec pos2)
{
    dvec temp = pos1 - pos2;
    pos2.x = temp.x < -0.5 ? pos2.x - 1.0 : (temp.x > 0.5 ? pos2.x + 1.0 : pos2.x);
    pos2.y = temp.y < -0.5 ? pos2.y - 1.0 : (temp.y > 0.5 ? pos2.y + 1.0 : pos2.y);
    pos2.z = temp.z < -0.5 ? pos2.z - 1.0 : (temp.z > 0.5 ? pos2.z + 1.0 : pos2.z);
    
    return pos1 - pos2;
}

__forceinline__ __device__
cubble::dvec cubble::getWrappedPos(dvec pos)
{
    // ASSUMPTION: Using normalized position
    // ASSUMPTION: Position never smaller/greater than -1/1
    pos.x = pos.x < 0 ? pos.x + 1.0 : (pos.x > 1 ? pos.x - 1.0 : pos.x);
    pos.y = pos.y < 0 ? pos.y + 1.0 : (pos.y > 1 ? pos.y - 1.0 : pos.y);
    pos.z = pos.z < 0 ? pos.z + 1.0 : (pos.z > 1 ? pos.z - 1.0 : pos.z);

    return pos;
}
