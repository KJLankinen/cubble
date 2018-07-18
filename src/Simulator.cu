// -*- C++ -*-

#include "Simulator.h"
#include "Macros.h"
#include "CudaContainer.h"
#include "Cell.h"
#include "Vec.h"
#include "Util.h"

#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

cubble::Simulator::~Simulator()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void cubble::Simulator::setupSimulation()
{
    generateBubbles();
    assignBubblesToCells(true);
}

void cubble::Simulator::integrate(bool useGasExchange, bool printTimings)
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    if (printTimings)
	std::cout << "Starting integration..." << std::endl;

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
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

    float predictionTime = 0.0f;
    float accelerationTime = 0.0f;
    float correctionTime = 0.0f;

    size_t numIntegrationSteps = 0;
    if (printTimings)
	std::cout << "\tStarting loop..." << std::endl;
    do
    {
	float elapsedTime = 0.0f;
	cudaEventRecord(start, 0);
	predict<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    tfr,
	    lbb,
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize(),
	    useGasExchange);
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	predictionTime += elapsedTime;

	cudaEventRecord(start, 0);
	accelerate<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    numberOfNeighbors.getDataPtr(),
	    neighborIndices.getDataPtr(),
	    energies.getDataPtr(),
	    tfr,
	    lbb,
	    bubbles.getSize(),
	    cells.getSize(),
	    neighborStride,
	    env->getFZeroPerMuZero(),
	    env->getKParameter(),
	    env->getPi(),
	    env->getMinRad(),
	    useGasExchange);
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	accelerationTime += elapsedTime;
	
	cudaEventRecord(start, 0);
	correct<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	    bubbles.getDataPtr(),
	    indices.getDataPtr(),
	    cells.getDataPtr(),
	    errors.getDataPtr(),
	    tfr,
	    lbb,
	    timeStep,
	    bubbles.getSize(),
	    cells.getSize(),
	    useGasExchange);
	
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	correctionTime += elapsedTime;
        
        error = *thrust::max_element(thrust::host,
					    errors.getDataPtr(),
					    errors.getDataPtr() + errors.getSize());
	
	if (error < env->getErrorTolerance() / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > env->getErrorTolerance())
	    timeStep *= 0.5;

	++numIntegrationSteps;
    }
    while (error > env->getErrorTolerance());
    
    if (printTimings)
	std::cout << "\tLoop done..." << std::endl;

    predictionTime /= numIntegrationSteps;
    accelerationTime /= numIntegrationSteps;
    correctionTime /= numIntegrationSteps;
    
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    ElasticEnergy = thrust::reduce(thrust::host,
				   energies.getDataPtr(),
				   energies.getDataPtr() + energies.getSize());

    CudaContainer<int> numBubblesToDelete(1);
    CudaContainer<int> toBeDeletedIndices(bubbles.getSize());

    if (printTimings)
	std::cout << "\tUpdating bubble data..." << std::endl;
    
    updateData<<<gridSize, numThreads, sizeof(Bubble) * maxNumBubbles>>>(
	bubbles.getDataPtr(),
	indices.getDataPtr(),
	cells.getDataPtr(),
	toBeDeletedIndices.getDataPtr(),
	numBubblesToDelete.getDataPtr(),
        bubbles.getSize(),
        cells.getSize(),
	env->getMinRad(),
	useGasExchange);
	
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    if (numBubblesToDelete[0] > 0)
    {
	std::cout << "\tRemoving " << numBubblesToDelete[0]
		  <<  " bubbles, since their radii are below the minimum radius."
		  << std::endl;
	
	std::vector<int> indicesToDelete;
	toBeDeletedIndices.dataToVec(indicesToDelete);
	indicesToDelete.resize(numBubblesToDelete[0]);

	std::sort(indicesToDelete.begin(),
		  indicesToDelete.end(),
		  [](int a, int b) { return a < b;});

	removeSmallBubbles(indicesToDelete);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (t2 - t1);
    
    if (printTimings)
	std::cout << "\tIntegration step average timings (ms):"
		  << "\n\tPrediction: " << predictionTime
		  << "\n\tAcceleration: " << accelerationTime
		  << "\n\tCorrection: " << correctionTime
		  << "\n\tTotal duration of function: " << duration.count() * 1000
		  << std::endl;

    ++integrationStep;

    if (integrationStep % 100 == 0)
	assignBubblesToCells();
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
						bubbles.getSize(),
						env->getPi());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    volume = thrust::reduce(thrust::host,
			    volumes.getDataPtr(),
			    volumes.getDataPtr() + volumes.getSize());
    
    return volume;
}

double cubble::Simulator::getAverageRadius() const
{
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    CudaContainer<double> radii(bubbles.getSize());
    int numThreads = 1024;
    int numBlocks = (int)std::ceil(bubbles.getSize() / (float)numThreads);

    getRadii<<<numBlocks, numThreads>>>(bubbles.getDataPtr(),
					radii.getDataPtr(),
					bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    double avgRad = thrust::reduce(thrust::host,
				   radii.getDataPtr(),
				   radii.getDataPtr() + radii.getSize());
    avgRad /= radii.getSize();
    
    return avgRad;
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
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    if (useVerboseOutput)
	std::cout << "Starting to assign bubbles to cells." << std::endl;
    
    const int numBubblesPerCell = env->getNumBubblesPerCell();
    dim3 gridSize = getGridSize(bubbles.getSize());
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec cellSize = (env->getTfr() - env->getLbb()) /
	dvec(gridSize.x, gridSize.y, gridSize.z);
    
    CudaContainer<double> radii(bubbles.getSize());
    int numThreads = 1024;
    int numBlocks = (int)std::ceil(bubbles.getSize() / (float)numThreads);

    getRadii<<<numBlocks, numThreads>>>(bubbles.getDataPtr(),
					radii.getDataPtr(),
					bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    double minCellSize = *thrust::max_element(thrust::host,
					      radii.getDataPtr(),
					      radii.getDataPtr() + radii.getSize());
    minCellSize *= 2.0;
    
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
    
    float elapsedTime = 0.0f;
    float offsetTime = 0.0f;
    float bubblesToCellsTime = 0.0f;
    float neighborTime = 0.0f;
    
    cudaEventRecord(start, 0);
    calculateOffsets<<<gridSize, numBubblesPerCell>>>(bubbles.getDataPtr(),
						      cells.getDataPtr(),
						      bubbles.getSize(),
						      cells.getSize());

    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    offsetTime += elapsedTime;

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
    
    cudaEventRecord(start, 0);
    bubblesToCells<<<gridSize,numBubblesPerCell>>>(bubbles.getDataPtr(),
						   indices.getDataPtr(),
						   cells.getDataPtr(),
						   bubbles.getSize());
    
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    bubblesToCellsTime += elapsedTime;
    
    numThreads = 512;
    int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;
    gridSize = getGridSize(bubbles.getSize());
    gridSize.z *= numDomains;
    assertGridSizeBelowLimit(gridSize);

    if (useVerboseOutput)
	std::cout << "\tFinding neighbors for each bubble..." << std::endl;
    
    cudaEventRecord(start, 0);
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
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    neighborTime += elapsedTime;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (t2 - t1);

    if (useVerboseOutput)
	std::cout << "\tReordering phase average timings (ms):"
		  << "\n\tOffset calculation: " << offsetTime
		  << "\n\tAssigning bubbles to cells: " << bubblesToCellsTime
		  << "\n\tFinding neighbors: " << neighborTime
		  << "\n\tTotal duration of function: " << duration.count() * 1000
		  << std::endl;
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

void cubble::Simulator::removeSmallBubbles(const std::vector<int> &indicesToDelete)
{   
    double radiusIncrement = 0;
    size_t origSize = bubbles.getSize();
    for (const auto &index : indicesToDelete)
    {
	double vol = 0;
	double radius = bubbles[index].getRadius();
	vol = radius * radius;
#if (NUM_DIM == 3)
	vol *= radius;
#endif
	radiusIncrement += vol;
	bubbles[index] = bubbles[bubbles.getSize() - 1];
	bubbles.popBack();
    }
    
    assert(bubbles.getSize() + indicesToDelete.size() == origSize);
    
    radiusIncrement /= bubbles.getSize();
    for (size_t i = 0; i < bubbles.getSize(); ++i)
    {
	double newRadius = bubbles[i].getRadius();
#if (NUM_DIM == 3)
	newRadius = newRadius * newRadius * newRadius;
	newRadius = std::cbrt(newRadius + radiusIncrement);
#else
	newRadius *= newRadius;
	newRadius = std::sqrt(newRadius + radiusIncrement);
#endif
	bubbles[i].setRadius(newRadius);
    }
    
    assignBubblesToCells(true);
}


// ******************************
// Kernels
// ******************************

__global__
void cubble::getRadii(Bubble *b, double *radii, int numBubbles)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
	radii[tid] = b[tid].getRadius();
}

__global__
void cubble::calculateVolumes(Bubble *b, double *volumes, int numBubbles, double pi)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double radius = b[tid].getRadius();
	double volume = radius * radius * pi;
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

	randomOffset = dvec::normalize(randomOffset) * avgRad * w[gid];
	randomOffset = (randomOffset - lbb) / (tfr - lbb);
	pos = getWrappedPos(pos + randomOffset);

	b[gid].setPos(pos);
	b[gid].setPosPred(pos);
	b[gid].setRadius(r[gid]);
	b[gid].setRadiusPred(r[gid]);
    }
}

__global__
void cubble::calculateOffsets(Bubble *bubbles,
			      Cell *cells,
			      int numBubbles,
			      int numCells)
{   
    int tid = getGlobalTid();
    
    if (tid < numBubbles)
    {
	dvec pos = bubbles[tid].getPos();
	ivec indexVec(gridDim.x * pos.x, gridDim.y * pos.y, gridDim.z * pos.z);
	int index = gridDim.x * gridDim.y * indexVec.z
	    + gridDim.x * indexVec.y
	    + indexVec.x;

	DEVICE_ASSERT(index < numCells);
	DEVICE_ASSERT(pos.x <= 1.0 && pos.y <= 1.0 && pos.z <= 1.0);
	
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
		     int numCells,
		     bool useGasExchange)
{
    extern __shared__ Bubble localBubbles[];

    int cid = blockIdx.z * gridDim.x * gridDim.y
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;

    DEVICE_ASSERT(cid < numCells);
    const Cell *self = &cells[cid];

    if (threadIdx.x < self->size)
    {
	const int gid = indices[self->offset + threadIdx.x];
	localBubbles[threadIdx.x] = bubbles[gid];
        Bubble *bubble = &localBubbles[threadIdx.x];

	const dvec interval = (tfr - lbb);
	dvec pos = lbb + bubble->getPos() * interval;
	pos += 0.5 * timeStep * (3.0 * bubble->getVel() - bubble->getVelPrev());
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);
	
	bubble->setPosPred(pos);

	if (useGasExchange)
	{
	    const double radius = bubble->getRadius() + 0.5 * timeStep
		* (3.0 * bubble->getRadiusChangeRate() - bubble->getRadiusChangeRatePrev());
	    
	    bubble->setRadiusPred(radius);
	}
	
	bubbles[gid] = localBubbles[threadIdx.x];
    }
}

__global__
void cubble::correct(Bubble *bubbles,
		     int *indices,
		     Cell *cells,
		     double *errors,
		     dvec tfr,
		     dvec lbb,
		     double timeStep,
		     int numBubbles,
		     int numCells,
		     bool useGasExchange)
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
        Bubble *bubble = &localBubbles[threadIdx.x];
        
	const dvec interval = (tfr - lbb);
	dvec pos = lbb + bubble->getPos() * interval;
	const dvec vel = bubble->getVel();
	const dvec velPred = bubble->getVelPred();
	pos += 0.5 * timeStep * (vel + velPred);
	
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);

	double radError = 0;
	if (useGasExchange)
	{
	    const double radius = bubble->getRadius() + 0.5 * timeStep
		* (bubble->getRadiusChangeRate() + bubble->getRadiusChangeRatePred());
	    
	    radError = radius - bubble->getRadiusPred();
	    radError = radError < 0 ? -radError : radError;
	    
	    bubble->setRadiusPred(radius);
	}
	
	double error = (pos - bubble->getPosPred()).getAbsolute().getMaxComponent();
	error = error > radError ? error : radError;
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
			double *energies,
			dvec tfr,
			dvec lbb,
			int numBubbles,
			int numCells,
			int neighborStride,
			double fZeroPerMuZero,
			double kParam,
			double pi,
			double minRad,
			bool useGasExchange)
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
        Bubble *bubble = &localBubbles[threadIdx.x];

	dvec acceleration(0, 0, 0);
	double energy = 0;
	double radiusChangeRate = 0;

	const double radius1 = bubble->getRadiusPred();
	if (radius1 > minRad)
	{
	    const double invRad1 = 1.0 / radius1;
	    for (int i = 0; i < numberOfNeighbors[gid]; ++i)
	    {
		const int index = neighborIndices[gid * neighborStride + i];
		DEVICE_ASSERT(index < numBubbles);
		const Bubble *neighbor = &bubbles[index];
		const double radius2 = neighbor->getRadiusPred();
		
		if (radius2 < minRad)
		    continue;
		
		const double radii = radius1 + radius2;
		dvec distance = getShortestWrappedNormalizedVec(bubble->getPosPred(),
								neighbor->getPosPred());
		distance *= (tfr - lbb);
		const double magnitude = distance.getLength();
		
		if (radii < magnitude)
		    continue;
		
		const double compressionDistance = radii - magnitude;
		const double relativeCompressionDistance = compressionDistance / radii;
		
		DEVICE_ASSERT(relativeCompressionDistance >= 0 && relativeCompressionDistance < 1.0);
	        
		acceleration += distance * relativeCompressionDistance / magnitude;
		energy += relativeCompressionDistance * compressionDistance;

		if (useGasExchange)
		{
		    double areaOfOverlap = 0;
		    if (magnitude < radius2 || magnitude < radius1)
		    {
			areaOfOverlap = radius1 < radius2 ? radius1 : radius2;
			areaOfOverlap *= areaOfOverlap;
		    }
		    else
		    {
			areaOfOverlap = radius2 * radius2
			    - radius1 * radius1
			    + magnitude * magnitude;
			areaOfOverlap /= 2.0 * magnitude;
			areaOfOverlap *= areaOfOverlap;
			areaOfOverlap = radius2 * radius2 - areaOfOverlap;
			areaOfOverlap = (areaOfOverlap > -0.000000001 && areaOfOverlap < 0)
			    ? -areaOfOverlap
			    : areaOfOverlap;
		    }
		    
		    DEVICE_ASSERT(areaOfOverlap >= 0);
		    
#if (NUM_DIM == 3)
		    areaOfOverlap *= pi;
#else
		    areaOfOverlap = 2.0 * sqrt(areaOfOverlap);
#endif
		    radiusChangeRate += areaOfOverlap * (1.0 / radius2 - invRad1);
		}
	    }
	}

	if (useGasExchange)
	    bubble->setRadiusChangeRatePred(radiusChangeRate * kParam);
	
	bubble->setVelPred(acceleration * fZeroPerMuZero);
	
	bubbles[gid] = localBubbles[threadIdx.x];
	energies[gid] = energy;
    }
}

__global__
void cubble::updateData(Bubble *bubbles,
			int *indices,
			Cell *cells,
			int *toBeDeletedIndices,
			int *numBubblesToDelete,
			int numBubbles,
			int numCells,
			double minRad,
			bool useGasExchange)
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
        Bubble *bubble = &localBubbles[threadIdx.x];

	bubble->setPos(bubble->getPosPred());
	bubble->setVelPrev(bubble->getVel());
	bubble->setVel(bubble->getVelPred());

	if (useGasExchange)
	{
	    bubble->setRadius(bubble->getRadiusPred());
	    bubble->setRadiusChangeRatePrev(bubble->getRadiusChangeRate());
	    bubble->setRadiusChangeRate(bubble->getRadiusChangeRatePred());
	}

	if (bubble->getRadius() < minRad)
	{
	    int index = atomicAdd(&numBubblesToDelete[0], 1);
	    DEVICE_ASSERT(index < numBubbles);
	    toBeDeletedIndices[index] = gid;
	}
	
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
