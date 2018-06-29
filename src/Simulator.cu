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
    
    int n = env->getNumBubbles();
    bubbles = CudaContainer<Bubble>(n);
}

cubble::Simulator::~Simulator()
{}

void cubble::Simulator::setupSimulation()
{
    generateBubbles();
    assignBubblesToCells(true);
    removeIntersectingBubbles();
    assignBubblesToCells(true);
}

void cubble::Simulator::integrate()
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    dvec tfr = env->getTfr();
    dvec lbb = env->getLbb();
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
	    bubbles.getDevPtr(),
	    indices.getDevPtr(),
	    cells.getDevPtr(),
	    tfr,
	    lbb,
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize());
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	// Calculate accelerations
	accelerations = CudaContainer<dvec>(bubbles.getSize());
	int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;
	numThreads = 512;
	gridSize = getGridSize(bubbles.getSize());
	gridSize.z *= numDomains;
	
	accelerate<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDevPtr(),
	    indices.getDevPtr(),
	    cells.getDevPtr(),
	    accelerations.getDevPtr(),
	    energies.getDevPtr(),
	    tfr,
	    lbb,
	    bubbles.getSize(),
	    numDomains,
	    cells.getSize(),
	    maxNumBubbles);
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	// Calculate predicted velocity, correction & error
	gridSize = getGridSize(bubbles.getSize());
	numThreads = maxNumBubbles;
	
	correct<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDevPtr(),
	    indices.getDevPtr(),
	    cells.getDevPtr(),
	    errors.getDevPtr(),
	    accelerations.getDevPtr(),
	    tfr,
	    lbb,
	    env->getFZeroPerMuZero(),
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize());
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	errors.deviceToHost();
        error = *thrust::max_element(thrust::host,
					    errors.getHostPtr(),
					    errors.getHostPtr() + errors.getSize());
	
	if (error < env->getErrorTolerance() / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > env->getErrorTolerance())
	    timeStep *= 0.5;
    }
    while (error > env->getErrorTolerance());

    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    energies.deviceToHost();
    ElasticEnergy = thrust::reduce(thrust::host,
				   energies.getHostPtr(),
				   energies.getHostPtr() + energies.getSize());

    // Update bubble data: current --> previous, predicted --> current
    gridSize = getGridSize(bubbles.getSize());
    numThreads = maxNumBubbles;
    updateData<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	bubbles.getDevPtr(),
	indices.getDevPtr(),
	cells.getDevPtr(),
        bubbles.getSize(),
        cells.getSize());
}

double cubble::Simulator::getVolumeOfBubbles() const
{
    // ASSUMPTION: Device data is up to date.
    double volume = 0;
    CudaContainer<double> volumes(bubbles.getSize());
    int numThreads = 1024;
    int numBlocks = (int)std::ceil(bubbles.getSize() / (float)numThreads);
    calculateVolumes<<<numBlocks, numThreads>>>(bubbles.getDevPtr(),
						volumes.getDevPtr(),
						bubbles.getSize());
    volumes.deviceToHost();
    volume = thrust::reduce(thrust::host,
			    volumes.getHostPtr(),
			    volumes.getHostPtr() + volumes.getSize());
    
    return volume;
}

void cubble::Simulator::getBubbles(std::vector<Bubble> &b) const
{ 
    // ASSUMPTION: Device data is up to date
    bubbles.deviceToVec(b);
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

    int numThreads = 1024;
    int numBlocks = (int)std::ceil(n / (float)numThreads);
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
    CudaContainer<float> r(n);

    // Generate random positions & radii
    std::cout << "\n\tGenerating data..." << std::flush;
    
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
						   n);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::cout << " Done.\n" << std::endl;
}

void cubble::Simulator::assignBubblesToCells(bool useVerboseOutput)
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    if (useVerboseOutput)
	std::cout << "Starting to assign bubbles to cells." << std::flush;
    
    int numBubblesPerCell = env->getNumBubblesPerCell();
    dim3 gridSize = getGridSize(bubbles.getSize());
    int numCells = gridSize.x * gridSize.y * gridSize.z;
    dvec cellSize = (env->getTfr() - env->getLbb()) /
	dvec(gridSize.x, gridSize.y, gridSize.z);
    double minCellSize = 3.0 * env->getAvgRad();
    
    if (useVerboseOutput)
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
    errors = CudaContainer<double>(bubbles.getSize());
    energies = CudaContainer<double>(bubbles.getSize());

    if (useVerboseOutput)
	std::cout << "\n\tCalculating offsets..." << std::flush;
    
    calculateOffsets<<<gridSize, numBubblesPerCell>>>(bubbles.getDevPtr(),
						      cells.getDevPtr(),
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
    
    if (useVerboseOutput)
	std::cout << " Done.\n\tAssigning bubbles to cells..." << std::flush;
    
    bubblesToCells<<<gridSize,numBubblesPerCell>>>(bubbles.getDevPtr(),
						   indices.getDevPtr(),
						   cells.getDevPtr(),
						   bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    cells.deviceToHost();
    
    if (useVerboseOutput)
	std::cout << " Done.\n" << std::endl;
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

    std::cout << "\tCalculating the size of dynamically allocated shared memory..."
	      << std::flush;

    
    int maxNumBubbles = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	int temp = cells[i].size;
	maxNumBubbles = maxNumBubbles < temp ? temp : maxNumBubbles;
    }
    
    // Nearest even size
    maxNumBubbles += maxNumBubbles % 2;
    assertMemBelowLimit(maxNumBubbles * sizeof(Bubble));

    std::cout << " Done.\n\tStarting kernel for finding intersections..." << std::flush;
    
    findIntersections<<<gridSize, numThreads, maxNumBubbles * sizeof(Bubble)>>>(
	bubbles.getDevPtr(),
	indices.getDevPtr(),
	cells.getDevPtr(),
	intersections.getDevPtr(),
	tfr,
	lbb,
	env->getInitialOverlapTolerance(),
	bubbles.getSize(),
	numDomains,
	cells.getSize(),
	maxNumBubbles);
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::cout << " Done.\n\tCopying values from device to host..." << std::flush;

    std::vector<Bubble> culledBubbles;
    bubbles.deviceToHost();
    intersections.deviceToHost();
    
    std::cout << " Done.\n\tRemoving intersecting elements from host vector..." << std::flush;

    assert(intersections.getSize() == bubbles.getSize());
    double minRadius = env->getMinRad();
    for (size_t i = 0; i < intersections.getSize(); ++i)
    {
	if (intersections[i] == 0)
	{
	    Bubble tempBubble = bubbles[i];
	    if (tempBubble.getRadius() >= minRadius)
		culledBubbles.push_back(tempBubble);
	}
    }

    std::cout << " Done.\n\tCopying data..." << std::flush;

    bubbles = CudaContainer<Bubble>(culledBubbles);
    
    std::cout << " Done.\n" << std::endl;
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
				 Bubble *b,
				 int numBubbles)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
#if (NUM_DIM == 3)
	dvec pos(x[tid], y[tid], z[tid]);
#else
	dvec pos(x[tid], y[tid], 0.0);
#endif  
	
	b[tid].setPos(pos);
	b[tid].setRadius((double)r[tid]);
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
void cubble::findIntersections(Bubble *bubbles,
			       int *indices,
			       Cell *cells,
			       int *intersectingIndices,
			       dvec tfr,
			       dvec lbb,
			       double overlapTolerance,
			       int numBubbles,
			       int numDomains,
			       int numCells,
			       int numLocalBubbles)
{
    extern __shared__ Bubble localBubbles[];

    deviceAssert(numBubbles > 0);
    deviceAssert(numDomains > 0);
    deviceAssert(numCells > 0);
    deviceAssert(numLocalBubbles > 0);
    deviceAssert(!(numDomains & 1));
    
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

    deviceAssert(xBegin >= 0 && xInterval > 0 && yBegin >= 0 && yInterval > 0);
    deviceAssert(xInterval + yInterval <= numLocalBubbles);
    
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
	    deviceAssert(y < yInterval);

	    int gid1 = xBegin + x;
	    int gid2 = yBegin + y;
	    
	    // Skip self-intersection
	    if (gid1 == gid2)
		continue;
	    
	    const Bubble *b1 = &localBubbles[x];
	    const Bubble *b2 = &localBubbles[xInterval + y];
	    
	    double radii = b1->getRadius() + b2->getRadius();
	    dvec posVec = getShortestWrappedNormalizedVec(b1->getPos(), b2->getPos());
	    double length = (posVec * (tfr - lbb)).getSquaredLength();
	    
	    if (radii * radii > overlapTolerance * length)
	    {
		gid1 = indices[gid1];
		gid2 = indices[gid2];
		int gid = gid1 < gid2 ? gid1 : gid2;
		
		atomicAdd(&intersectingIndices[gid], 1);
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

    deviceAssert(cid < numCells);
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

    deviceAssert(cid < numCells);
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
		        dvec *accelerations,
			double *energies,
			dvec tfr,
			dvec lbb,
			int numBubbles,
			int numDomains,
			int numCells,
			int numLocalBubbles)
{
    extern __shared__ Bubble localBubbles[];

    deviceAssert(numBubbles > 0);
    deviceAssert(numDomains > 0);
    deviceAssert(numCells > 0);
    deviceAssert(numLocalBubbles > 0);
    deviceAssert(!(numDomains & 1));
    
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

    deviceAssert(xBegin >= 0 && xInterval > 0 && yBegin >= 0 && yInterval > 0);
    deviceAssert(xInterval + yInterval <= numLocalBubbles);
    
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
    double accelerationMultiplier = isOwnCell ? 0.5 : 1.0;

    for (int round = 0; round < numRounds; ++round)
    {
        int pairIdx = round * blockDim.x + threadIdx.x;
	if (pairIdx < numPairs)
	{
	    int x = pairIdx % xInterval;
	    int y = pairIdx / xInterval;
	    deviceAssert(y < yInterval);

	    int gid1 = indices[xBegin + x];
	    int gid2 = indices[yBegin + y];
	    
	    // Skip self-intersection
	    if (gid1 == gid2)
		continue;
	    
	    const Bubble *b1 = &localBubbles[x];
	    const Bubble *b2 = &localBubbles[xInterval + y];
	    
	    double radii = b1->getRadius() + b2->getRadius();
	    dvec acceleration = getShortestWrappedNormalizedVec(
		b1->getPosPred(),
		b2->getPosPred());
	    acceleration *= (tfr - lbb);
	    
	    if (radii * radii > acceleration.getSquaredLength())
	    {
		double magnitude = acceleration.getLength();
		double temp = radii - magnitude;
		acceleration *= temp / (radii * magnitude);
		acceleration *= accelerationMultiplier;
		double energy = (temp * temp) / radii;

		// HACK: This is wiiiiiildly inefficient.
		atomicAddD(&accelerations[gid1].x, acceleration.x);
		atomicAddD(&accelerations[gid1].y, acceleration.y);
		atomicAddD(&accelerations[gid1].z, acceleration.z);
		atomicAddD(&accelerations[gid2].x, -acceleration.x);
		atomicAddD(&accelerations[gid2].y, -acceleration.y);
		atomicAddD(&accelerations[gid2].z, -acceleration.z);
		atomicAddD(&energies[gid1], energy);
	    }
	}
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

    deviceAssert(cid < numCells);
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
    
    deviceAssert((di == 0 && domain < (int)(0.5f * numDomains))
	   || (di == 1 && domain >= (int)(0.5f * numDomains)));
    
    int dj = domain % (int)(0.5f * numDomains);
    int djMod2 = dj % 2;

    // Find this cell
    int selfCellIndex = cellIdxVec.z * boxDim.x * boxDim.y
	+ cellIdxVec.y * boxDim.x
	+ cellIdxVec.x;
    deviceAssert(selfCellIndex < numCells);
    Cell self = cells[selfCellIndex];

    // Find the neighbor of this cell
    int neighborCellIndex = getNeighborCellIndex(cellIdxVec, boxDim, dj / 2);
    deviceAssert(neighborCellIndex < numCells);
    Cell neighbor = cells[neighborCellIndex];
    
    outIsOwnCell = selfCellIndex == neighborCellIndex;

    // Find the interval of values to use:
    // x-axis uses the right or the left half of the neighbor cell
    int halfSize = 0.5f * neighbor.size;
    outXBegin = neighbor.offset + djMod2 * halfSize;
    outXInterval = halfSize + djMod2 * (neighbor.size % 2);
    
    deviceAssert(outXBegin + outXInterval <= numBubbles);
    deviceAssert(outXBegin + outXInterval <= neighbor.size + neighbor.offset);
    deviceAssert(outXInterval == halfSize || outXInterval == halfSize + 1);

    // y-axis uses the top or bottom half of this cell
    halfSize = 0.5f * self.size;
    outYBegin = self.offset + di * halfSize;
    outYInterval = halfSize + di * (self.size % 2);

    deviceAssert(outYBegin + outYInterval <= numBubbles);
    deviceAssert(outYInterval == halfSize || outYInterval == halfSize + 1);
    deviceAssert(outYBegin + outYInterval <= self.size + self.offset);
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
