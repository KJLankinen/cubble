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
#include "cub/cub/cub.cuh"


// ******************************
// Class functions run on CPU
// ******************************

cubble::Simulator::Simulator(std::shared_ptr<Env> e)
{
    env = e;
    
#if (NUM_DIM == 3)
    givenNumBubblesPerDim = std::ceil(std::cbrt((float)env->getNumBubbles()));
    numBubbles = givenNumBubblesPerDim * givenNumBubblesPerDim * givenNumBubblesPerDim;
#else
    givenNumBubblesPerDim = std::ceil(std::sqrt((float)env->getNumBubbles()));
    numBubbles = givenNumBubblesPerDim * givenNumBubblesPerDim;
#endif
    const dvec tfr = env->getLbb() + env->getAvgRad() * (double)givenNumBubblesPerDim * 2;
    env->setTfr(tfr);
    
    dmh = std::unique_ptr<cubble::DeviceMemoryHandler>(new DeviceMemoryHandler(numBubbles));
    dmh->reserveMemory();
    
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

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton perdictor-corrector method (ABMpc).

    double *x = dmh->getDataPtr(BubbleProperty::X);
    double *y = dmh->getDataPtr(BubbleProperty::Y);
    double *z = dmh->getDataPtr(BubbleProperty::Z);
    double *r = dmh->getDataPtr(BubbleProperty::R);
    
    double *dxdt = dmh->getDataPtr(BubbleProperty::DXDT);
    double *dydt = dmh->getDataPtr(BubbleProperty::DYDT);
    double *dzdt = dmh->getDataPtr(BubbleProperty::DZDT);
    double *drdt = dmh->getDataPtr(BubbleProperty::DRDT);
    
    double *dxdtOld = dmh->getDataPtr(BubbleProperty::DXDT_OLD);
    double *dydtOld = dmh->getDataPtr(BubbleProperty::DYDT_OLD);
    double *dzdtOld = dmh->getDataPtr(BubbleProperty::DZDT_OLD);
    double *drdtOld = dmh->getDataPtr(BubbleProperty::DRDT_OLD);
    
    double *energies = dmh->getDataPtr(BubbleProperty::ENERGY);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    const size_t numThreads = 512;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);

    double timeStep = env->getTimeStep();

    accelerate<<<numBlocks, numThreads>>>(x, y, z, r,
					  dxdtOld, dydtOld, dzdtOld, drdtOld,
					  energies,
					  numberOfNeighbors.getDataPtr(),
					  neighborIndices.getDataPtr(),
					  tfr,
					  lbb,
					  numBubbles,
					  neighborStride,
					  env->getFZeroPerMuZero(),
					  env->getKParameter(),
					  env->getPi(),
					  env->getMinRad(),
					  true);

    eulerIntegration<<<numBlocks, numThreads>>>(x, y, z, r,
						dxdtOld, dydtOld, dzdtOld, drdtOld,
						tfr, lbb, timeStep, numBubbles);
    
    accelerate<<<numBlocks, numThreads>>>(x, y, z, r,
					  dxdt, dydt, dzdt, drdt,
					  energies,
					  numberOfNeighbors.getDataPtr(),
					  neighborIndices.getDataPtr(),
					  tfr,
					  lbb,
					  numBubbles,
					  neighborStride,
					  env->getFZeroPerMuZero(),
					  env->getKParameter(),
					  env->getPi(),
					  env->getMinRad(),
					  true);
}

void cubble::Simulator::integrate(bool useGasExchange)
{
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    const size_t numThreads = 512;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);

    double timeStep = env->getTimeStep();
    double error = 0;

    size_t numIntegrationSteps = 0;

    double *x = dmh->getDataPtr(BubbleProperty::X);
    double *y = dmh->getDataPtr(BubbleProperty::Y);
    double *z = dmh->getDataPtr(BubbleProperty::Z);
    double *r = dmh->getDataPtr(BubbleProperty::R);
    
    double *xPrd = dmh->getDataPtr(BubbleProperty::X_PRD);
    double *yPrd = dmh->getDataPtr(BubbleProperty::Y_PRD);
    double *zPrd = dmh->getDataPtr(BubbleProperty::Z_PRD);
    double *rPrd = dmh->getDataPtr(BubbleProperty::R_PRD);
    
    double *dxdt = dmh->getDataPtr(BubbleProperty::DXDT);
    double *dydt = dmh->getDataPtr(BubbleProperty::DYDT);
    double *dzdt = dmh->getDataPtr(BubbleProperty::DZDT);
    double *drdt = dmh->getDataPtr(BubbleProperty::DRDT);
    
    double *dxdtPrd = dmh->getDataPtr(BubbleProperty::DXDT_PRD);
    double *dydtPrd = dmh->getDataPtr(BubbleProperty::DYDT_PRD);
    double *dzdtPrd = dmh->getDataPtr(BubbleProperty::DZDT_PRD);
    double *drdtPrd = dmh->getDataPtr(BubbleProperty::DRDT_PRD);
    
    double *dxdtOld = dmh->getDataPtr(BubbleProperty::DXDT_OLD);
    double *dydtOld = dmh->getDataPtr(BubbleProperty::DYDT_OLD);
    double *dzdtOld = dmh->getDataPtr(BubbleProperty::DZDT_OLD);
    double *drdtOld = dmh->getDataPtr(BubbleProperty::DRDT_OLD);

    double *energies = dmh->getDataPtr(BubbleProperty::ENERGY);
    double *errors = dmh->getDataPtr(BubbleProperty::ERROR);
    
    do
    {
	predict<<<numBlocks, numThreads>>>(x, y, z, r,
					   xPrd, yPrd, zPrd, rPrd,
					   dxdt, dydt, dzdt, drdt,
					   dxdtOld, dydtOld, dzdtOld, drdtOld,
					   tfr, lbb, timeStep, numBubbles, useGasExchange);
        
	accelerate<<<numBlocks, numThreads>>>(xPrd, yPrd, zPrd, rPrd,
					      dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
					      energies,
					      numberOfNeighbors.getDataPtr(),
					      neighborIndices.getDataPtr(),
					      tfr,
					      lbb,
					      numBubbles,
					      neighborStride,
					      env->getFZeroPerMuZero(),
					      env->getKParameter(),
					      env->getPi(),
					      env->getMinRad(),
					      useGasExchange);
        
	correct<<<numBlocks, numThreads>>>(x, y, z, r,
					   xPrd, yPrd, zPrd, rPrd,
					   dxdt, dydt, dzdt, drdt,
					   dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
					   errors,
					   tfr,
					   lbb,
					   timeStep,
					   numBubbles,
					   useGasExchange);
        
        error = getMaxElement(errors, numBubbles);
	
	if (error < env->getErrorTolerance() / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > env->getErrorTolerance())
	    timeStep *= 0.5;

	++numIntegrationSteps;
    }
    while (error > env->getErrorTolerance());
    
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    ElasticEnergy = sumReduction(energies, numBubbles);
    
    updateData<<<numBlocks, numThreads>>>(x, y, z, r,
					  xPrd, yPrd, zPrd, rPrd,
					  dxdt, dydt, dzdt, drdt,
					  dxdtOld, dydtOld, dzdtOld, drdtOld,
					  dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
					  toBeDeletedIndices.getDataPtr(),
					  numBubblesToDelete.getDataPtr(),
					  numBubbles,
					  env->getMinRad(),
					  useGasExchange);
    
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());
    
    if (numBubblesToDelete[0] > 0)
    {
	std::cout << "Should remove small bubble(s)..." << std::endl;
	/*
	std::cout << "\tRemoving " << numBubblesToDelete[0]
		  <<  " bubbles, since their radii are below the minimum radius."
		  << std::endl;
	
	std::vector<int> indicesToDelete;
	toBeDeletedIndices.dataToVec(indicesToDelete);
	indicesToDelete.resize(numBubblesToDelete[0]);

	std::sort(indicesToDelete.begin(), indicesToDelete.end(), [](int a, int b) { return a < b; } );

	removeSmallBubbles(indicesToDelete);
	*/
    }

    ++integrationStep;

    if (integrationStep % 100 == 0)
	assignBubblesToCells();
}

double cubble::Simulator::getVolumeOfBubbles() const
{
    const size_t numThreads = 512;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);

    double *volPtr = dmh->getDataPtr(BubbleProperty::VOLUME);
    
    calculateVolumes<<<numBlocks, numThreads>>>(
	dmh->getDataPtr(BubbleProperty::R), volPtr, numBubbles, env->getPi());
    
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());

    double volume = sumReduction(volPtr, numBubbles);
    
    return volume;
}

double cubble::Simulator::getAverageRadius() const
{
    double *r = dmh->getDataPtr(BubbleProperty::R);
    double avgRad = sumReduction(r, numBubbles);
    avgRad/= numBubbles;
    
    return avgRad;
}

void cubble::Simulator::getBubbles(std::vector<Bubble> &bubbles) const
{
    // Very inefficient function, should rework.
    bubbles.clear();
    bubbles.resize(numBubbles);

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> r;
    x.resize(numBubbles);
    y.resize(numBubbles);
    z.resize(numBubbles);
    r.resize(numBubbles);

    double *devX = dmh->getDataPtr(BubbleProperty::X);
    double *devY = dmh->getDataPtr(BubbleProperty::Y);
    double *devZ = dmh->getDataPtr(BubbleProperty::Z);
    double *devR = dmh->getDataPtr(BubbleProperty::R);

    cudaMemcpy(x.data(), devX, sizeof(double) * numBubbles, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), devY, sizeof(double) * numBubbles, cudaMemcpyDeviceToHost);
    cudaMemcpy(z.data(), devZ, sizeof(double) * numBubbles, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data(), devR, sizeof(double) * numBubbles, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numBubbles; ++i)
    {
	Bubble b;
	dvec pos(x[i], y[i], z[i]);
	b.setPos(pos);
	b.setRadius(r[i]);
	bubbles[i] = b;
    }
}

double cubble::Simulator::sumReduction(double *inputData, size_t lengthOfData) const
{
    // Stupid and inefficient, but it's just temporary
    double *outputData = nullptr;
    void *devTempStoragePtr = NULL;
    size_t tempStorageBytes = 0;
    std::vector<double> stupidVec(1);
    
    cub::DeviceReduce::Sum(devTempStoragePtr, tempStorageBytes, inputData, outputData, lengthOfData);

    cudaMalloc(&devTempStoragePtr, tempStorageBytes);
    cudaMalloc((void**)&outputData, sizeof(double));
    
    cub::DeviceReduce::Sum(devTempStoragePtr, tempStorageBytes, inputData, outputData, lengthOfData);
    
    cudaMemcpy(stupidVec.data(), outputData, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(outputData);
    cudaFree(devTempStoragePtr);

    return stupidVec[0];
}

double cubble::Simulator::getMaxElement(double *inputData, size_t lengthOfData) const
{
    // Stupid and inefficient, but it's just temporary
    double *outputData = nullptr;
    void *devTempStoragePtr = NULL;
    size_t tempStorageBytes = 0;
    std::vector<double> stupidVec(1);
    
    cub::DeviceReduce::Max(devTempStoragePtr, tempStorageBytes, inputData, outputData, lengthOfData);

    cudaMalloc(&devTempStoragePtr, tempStorageBytes);
    cudaMalloc((void**)&outputData, sizeof(double));
    
    cub::DeviceReduce::Max(devTempStoragePtr, tempStorageBytes, inputData, outputData, lengthOfData);
    
    cudaMemcpy(stupidVec.data(), outputData, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(outputData);
    cudaFree(devTempStoragePtr);

    return stupidVec[0];   
}

void cubble::Simulator::generateBubbles()
{
    std::cout << "Starting to generate data for bubbles." << std::endl;
    
    const int rngSeed = env->getRngSeed();
    const double avgRad = env->getAvgRad();
    const double stdDevRad = env->getStdDevRad();
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();

    indices = CudaContainer<int>(numBubbles);
    numberOfNeighbors = CudaContainer<int>(numBubbles);
    neighborIndices = CudaContainer<int>(numBubbles * neighborStride);
    numBubblesToDelete = CudaContainer<int>(1);
    toBeDeletedIndices = CudaContainer<int>(numBubbles);
    
    std::cout << "\tGenerating data..." << std::endl;

    double *x = dmh->getDataPtr(BubbleProperty::X);
    double *y = dmh->getDataPtr(BubbleProperty::Y);
    double *z = dmh->getDataPtr(BubbleProperty::Z);
    
    double *xPrd = dmh->getDataPtr(BubbleProperty::X_PRD);
    double *yPrd = dmh->getDataPtr(BubbleProperty::Y_PRD);
    double *zPrd = dmh->getDataPtr(BubbleProperty::Z_PRD);
    
    double *r = dmh->getDataPtr(BubbleProperty::R);
    double *w = dmh->getDataPtr(BubbleProperty::R_PRD);
    
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    
    CURAND_CALL(curandGenerateUniformDouble(generator, x, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, y, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, z, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, r, numBubbles, avgRad, stdDevRad));

    CURAND_CALL(curandDestroyGenerator(generator));

    std::cout << "\tAssigning data to bubbles..." << std::endl;;

    const size_t numThreads = 512;
    const size_t numBlocks = (size_t)std::ceil((float)numBubbles / (float)numThreads);
    assignDataToBubbles<<<numBlocks, numThreads>>>(x, y, z,
						   xPrd, yPrd, zPrd,
						   r, w, givenNumBubblesPerDim, tfr, lbb, avgRad, numBubbles);
}

void cubble::Simulator::assignBubblesToCells(bool useVerboseOutput)
{
    if (useVerboseOutput)
	std::cout << "Starting to assign bubbles to cells." << std::endl;
    
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec domainDim(gridSize.x, gridSize.y, gridSize.z);
    const dvec cellSize = (env->getTfr() - env->getLbb()) / domainDim;
    const size_t numThreads = 512;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    const int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;

    double *x = dmh->getDataPtr(BubbleProperty::X);
    double *y = dmh->getDataPtr(BubbleProperty::Y);
    double *z = dmh->getDataPtr(BubbleProperty::Z);
    double *r = dmh->getDataPtr(BubbleProperty::R);

    const double minCellSize = getMaxElement(r, numBubbles);
    
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

    cudaMemset((void*)numBubblesToDelete.getDataPtr(), 0, sizeof(int));
    cudaMemset((void*)numberOfNeighbors.getDataPtr(), 0, sizeof(int) * numBubbles);
    cudaMemset((void*)indices.getDataPtr(), 0, sizeof(int) * numBubbles);
    cudaMemset((void*)neighborIndices.getDataPtr(), 0, sizeof(int) * numBubbles * neighborStride);
    
    dmh->resetData(BubbleProperty::ERROR);
    dmh->resetData(BubbleProperty::ENERGY);
    dmh->resetData(BubbleProperty::VOLUME);
    
    if (useVerboseOutput)
	std::cout << "\tCalculating offsets..." << std::endl;
    
    calculateOffsets<<<numBlocks, numThreads>>>(x, y, z, cells.getDataPtr(), domainDim, numBubbles, numCells);

    CUDA_CALL(cudaDeviceSynchronize());
    int cumulativeSum = 0;
    for (size_t i = 0; i < cells.getSize(); ++i)
    {
	const int numBubbles = cells[i].offset;
	cells[i].offset = cumulativeSum;
	cumulativeSum += numBubbles;
    }
    
    if (useVerboseOutput)
	std::cout << "\tAssigning bubbles to cells..." << std::endl;

    bubblesToCells<<<numBlocks, numThreads>>>(
	x, y, z, indices.getDataPtr(), cells.getDataPtr(), domainDim, numBubbles);

    gridSize.z *= numDomains;
    assertGridSizeBelowLimit(gridSize);

    if (useVerboseOutput)
	std::cout << "\tFinding neighbors for each bubble..." << std::endl;
    
    findNeighbors<<<gridSize, numThreads>>>(x, y, z, r,
					    indices.getDataPtr(),
					    cells.getDataPtr(),
					    numberOfNeighbors.getDataPtr(),
					    neighborIndices.getDataPtr(),
					    env->getTfr(),
					    env->getLbb(),
					    numBubbles,
					    numDomains,
					    cells.getSize(),
					    neighborStride);
}

dim3 cubble::Simulator::getGridSize()
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
void cubble::calculateVolumes(double *r, double *volumes, int numBubbles, double pi)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double radius = r[tid];
	double volume = radius * radius * pi;
#if (NUM_DIM == 3)
	volume *= radius * 1.33333333333333333333333333;
#endif
	
	volumes[tid] = volume;
    }   
}

__global__
void cubble::assignDataToBubbles(double *x,
				 double *y,
				 double *z,
				 double *xPrd,
				 double *yPrd,
				 double *zPrd,
				 double *r,
				 double *w,
				 int givenNumBubblesPerDim,
				 dvec tfr,
				 dvec lbb,
				 double avgRad,
				 int numBubbles)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	int xid = tid % givenNumBubblesPerDim;
	int yid = (tid / givenNumBubblesPerDim) % givenNumBubblesPerDim;
	
	dvec randomOffset(x[tid], y[tid], 0);
	dvec pos(0, 0, 0);
	pos.x = xid / (double)givenNumBubblesPerDim;
	pos.y = yid / (double)givenNumBubblesPerDim;
#if (NUM_DIM == 3)
	int zid = tid / (givenNumBubblesPerDim * givenNumBubblesPerDim);
	pos.z = zid / (double)givenNumBubblesPerDim;
	randomOffset.z = z[tid];
#endif

	randomOffset = dvec::normalize(randomOffset) * avgRad * w[tid];
	randomOffset = (randomOffset - lbb) / (tfr - lbb);
	pos = getWrappedPos(pos + randomOffset);

	x[tid] = pos.x;
	y[tid] = pos.y;
	z[tid] = pos.z;
	
	xPrd[tid] = pos.x;
	yPrd[tid] = pos.y;
	zPrd[tid] = pos.z;
	
	w[tid] = r[tid];
    }
}

__global__
void cubble::calculateOffsets(double *x,
			      double *y,
			      double *z,
			      Cell *cells,
			      dvec domainDim,
			      int numBubbles,
			      int numCells)
{   
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	dvec pos = dvec(0, 0, 0);
	pos.x = x[tid];
	pos.y = y[tid];
	pos.z = z[tid];
	
        const ivec indexVec = (pos * domainDim).asType<int>();
	const int index = domainDim.x * domainDim.y * indexVec.z + domainDim.x * indexVec.y + indexVec.x;
	DEVICE_ASSERT(index < numCells);
	
	atomicAdd(&cells[index].offset, 1);
    }
}

__global__
void cubble::bubblesToCells(double *x,
			    double *y,
			    double *z,
			    int *indices,
			    Cell *cells,
			    dvec domainDim,
			    int numBubbles)
{
    int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	dvec pos = dvec(0, 0, 0);
	pos.x = x[tid];
	pos.y = y[tid];
	pos.z = z[tid];
	
        const ivec indexVec = (pos * domainDim).asType<int>();
	const int index = domainDim.x * domainDim.y * indexVec.z + domainDim.x * indexVec.y + indexVec.x;
	const int offset = cells[index].offset + atomicAdd(&cells[index].size, 1);
        indices[offset] = tid;
    }
}

__global__
void cubble::findNeighbors(double *x,
			   double *y,
			   double *z,
			   double *r,
			   int *indices,
			   Cell *cells,
			   int *numberOfNeighbors,
			   int *neighborIndices,
			   dvec tfr,
			   dvec lbb,
			   int numBubbles,
			   int numDomains,
			   int numCells,
			   int neighborStride)
{
    DEVICE_ASSERT(numBubbles > 0);
    DEVICE_ASSERT(numDomains > 0);
    DEVICE_ASSERT(numCells > 0);
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
    
    int numPairs = xInterval * yInterval;
    int numRounds = 1 + (numPairs / blockDim.x);
    
    for (int round = 0; round < numRounds; ++round)
    {
        int pairIdx = round * blockDim.x + threadIdx.x;
	if (pairIdx < numPairs)
	{
	    int xid = pairIdx % xInterval;
	    int yid = pairIdx / xInterval;
	    DEVICE_ASSERT(yid < yInterval);
	    
	    int gid1 = indices[xBegin + xid];
	    int gid2 = indices[yBegin + yid];
	    
	    if (gid1 == gid2)
		continue;

	    dvec pos1, pos2;
	    pos1.x = x[gid1];
	    pos1.y = y[gid1];
	    pos1.z = z[gid1];
	    
	    pos2.x = x[gid2];
	    pos2.y = y[gid2];
	    pos2.z = z[gid2];
	    
	    dvec posVec = getShortestWrappedNormalizedVec(pos1, pos2);
	    const double length = (posVec * (tfr - lbb)).getSquaredLength();
	    
	    const double radii = r[gid1] + r[gid2];
	    
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
void cubble::predict(double *x,
		     double *y,
		     double *z,
		     double *r,
		     
		     double *xPrd,
		     double *yPrd,
		     double *zPrd,
		     double *rPrd,
		     
		     double *dxdt,
		     double *dydt,
		     double *dzdt,
		     double *drdt,
		     
		     double *dxdtOld,
		     double *dydtOld,
		     double *dzdtOld,
		     double *drdtOld,
		     
		     dvec tfr,
		     dvec lbb,
		     double timeStep,
		     int numBubbles,
		     bool useGasExchange)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	// Measure if it's faster to calculate these per component...
	const dvec interval = (tfr - lbb);
	dvec pos, vel, velOld;
	pos.x = x[tid];
	pos.y = y[tid];
	pos.z = z[tid];
	
	vel.x = dxdt[tid];
	vel.y = dydt[tid];
	vel.z = dzdt[tid];
	
	velOld.x = dxdtOld[tid];
	velOld.y = dydtOld[tid];
	velOld.z = dzdtOld[tid];

	pos = lbb + pos * interval;
	pos += 0.5 * timeStep * (3.0 * vel - velOld);
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);

	xPrd[tid] = pos.x;
	yPrd[tid] = pos.y;
	zPrd[tid] = pos.z;

	if (useGasExchange)
	    rPrd[tid] = r[tid] + 0.5 * timeStep * (3.0 * drdt[tid] - drdtOld[tid]);
    }
}

__global__
void cubble::accelerate(double *x,
			double *y,
			double *z,
			double *r,
			
			double *dxdt,
			double *dydt,
			double *dzdt,
			double *drdt,
			
			double *energies,
			int *numberOfNeighbors,
			int *neighborIndices,
			dvec tfr,
			dvec lbb,
			int numBubbles,
			int neighborStride,
			double fZeroPerMuZero,
			double kParam,
			double pi,
			double minRad,
			bool useGasExchange)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	dvec acceleration(0, 0, 0);
	double energy = 0;
	double radiusChangeRate = 0;

	const double radius1 = r[tid];
        dvec pos1(0, 0, 0);
	pos1.x = x[tid];
	pos1.y = y[tid];
	pos1.z = z[tid];
	
	if (radius1 > minRad)
	{
	    const double invRad1 = 1.0 / radius1;
	    for (int i = 0; i < numberOfNeighbors[tid]; ++i)
	    {
		const int index = neighborIndices[tid * neighborStride + i];
		DEVICE_ASSERT(index < numBubbles);
		
		const double radius2 = r[index];
	        dvec pos2(0, 0, 0);
		pos2.x = x[index];
		pos2.y = y[index];
		pos2.z = z[index];
		
		if (radius2 < minRad)
		    continue;
		
		const double radii = radius1 + radius2;
		dvec distance = getShortestWrappedNormalizedVec(pos1, pos2);
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
	    drdt[tid] = radiusChangeRate * kParam;
	
	dxdt[tid] = acceleration.x * fZeroPerMuZero;
	dydt[tid] = acceleration.y * fZeroPerMuZero;
	dzdt[tid] = acceleration.z * fZeroPerMuZero;
	energies[tid] = energy;
    }
}

__global__
void cubble::correct(double *x,
		     double *y,
		     double *z,
		     double *r,
		     
		     double *xPrd,
		     double *yPrd,
		     double *zPrd,
		     double *rPrd,
		     
		     double *dxdt,
		     double *dydt,
		     double *dzdt,
		     double *drdt,
		     
		     double *dxdtPrd,
		     double *dydtPrd,
		     double *dzdtPrd,
		     double *drdtPrd,
		     
		     double *errors,
		     dvec tfr,
		     dvec lbb,
		     double timeStep,
		     int numBubbles,
		     bool useGasExchange)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {   
	// Measure if it's faster to calculate these per component...
	const dvec interval = (tfr - lbb);
	dvec pos, posPrd, vel, velPrd;
	pos.x = x[tid];
	pos.y = y[tid];
	pos.z = z[tid];

	posPrd.x = xPrd[tid];
	posPrd.y = yPrd[tid];
	posPrd.z = zPrd[tid];
	
	vel.x = dxdt[tid];
	vel.y = dydt[tid];
	vel.z = dzdt[tid];
	
	velPrd.x = dxdtPrd[tid];
	velPrd.y = dydtPrd[tid];
	velPrd.z = dzdtPrd[tid];

	pos = lbb + pos * interval;
	pos += 0.5 * timeStep * (vel + velPrd);
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);

	double radError = 0;
	if (useGasExchange)
	{
	    const double radius = r[tid] + 0.5 * timeStep
		* (drdt[tid] + drdtPrd[tid]);
	    
	    radError = radius - rPrd[tid];
	    radError = radError < 0 ? -radError : radError;

	    rPrd[tid] = radius;
	}

	double error = (pos - posPrd).getAbsolute().getMaxComponent();
	error = error > radError ? error : radError;
	errors[tid] = error;

	xPrd[tid] = pos.x;
	yPrd[tid] = pos.y;
	zPrd[tid] = pos.z;
    }
}

__global__
void cubble::updateData(double *x,
			double *y,
			double *z,
			double *r,
			
			double *xPrd,
			double *yPrd,
			double *zPrd,
			double *rPrd,
			
			double *dxdt,
			double *dydt,
			double *dzdt,
			double *drdt,
			
			double *dxdtOld,
			double *dydtOld,
			double *dzdtOld,
			double *drdtOld,
			
			double *dxdtPrd,
			double *dydtPrd,
			double *dzdtPrd,
			double *drdtPrd,
			
			int *toBeDeletedIndices,
			int *numBubblesToDelete,
			int numBubbles,
			double minRad,
			bool useGasExchange)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	x[tid] = xPrd[tid];
	y[tid] = yPrd[tid];
	z[tid] = zPrd[tid];

	dxdtOld[tid] = dxdt[tid];
	dydtOld[tid] = dydt[tid];
	dzdtOld[tid] = dzdt[tid];

        dxdt[tid] = dxdtPrd[tid];
	dydt[tid] = dydtPrd[tid];
	dzdt[tid] = dzdtPrd[tid];

	if (useGasExchange)
	{
	    r[tid] = rPrd[tid];
	    drdtOld[tid] = drdt[tid];
	    drdt[tid] = drdtPrd[tid];
	}

	if (r[tid] < minRad)
	{
	    int index = atomicAdd(&numBubblesToDelete[0], 1);
	    DEVICE_ASSERT(index < numBubbles);
	    toBeDeletedIndices[index] = tid;
	}
    }
}

__global__
void cubble::eulerIntegration(double *x,
			      double *y,
			      double *z,
			      double *r,
			      
			      double *dxdt,
			      double *dydt,
			      double *dzdt,
			      double *drdt,
			      
			      dvec tfr,
			      dvec lbb,
			      double timeStep,
			      int numBubbles)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	dvec interval = tfr - lbb;
	dvec pos(0, 0, 0);
	pos.x = x[tid];
	pos.y = y[tid];
	pos.z = z[tid];

	dvec vel(0, 0, 0);
	vel.x = dxdt[tid];
	vel.y = dydt[tid];
	vel.z = dzdt[tid];

	pos = lbb + pos * interval;
	pos += timeStep * vel;
	pos = (pos - lbb) / interval;
	pos = getWrappedPos(pos);
	
	x[tid] = pos.x;
	y[tid] = pos.y;
	z[tid] = pos.z;
	r[tid] = r[tid] + timeStep * drdt[tid];
    }
}

// ******************************
// Device functions
// ******************************

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
