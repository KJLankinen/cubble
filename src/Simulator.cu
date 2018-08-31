// -*- C++ -*-

#include "Simulator.h"
#include "Macros.h"
#include "Vec.h"
#include "Util.h"

#include "cub/cub/cub.cuh"

#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <curand.h>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>


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

    bubbleData = FixedSizeDeviceArray<double>(numBubbles, (size_t)BubbleProperty::NUM_VALUES);

    indicesPerCell = FixedSizeDeviceArray<int>(numBubbles, 1);
    // TODO: Figure out a more sensible value for this.
    const int maxNumPairs = (CUBBLE_NUM_NEIGHBORS + 1) * env->getNumBubblesPerCell() * numBubbles;
    neighborPairIndices = FixedSizeDeviceArray<int>(maxNumPairs, 4);
    numPairs = FixedSizeDeviceArray<int>(1, 1);
    
    const dim3 gridSize = getGridSize();
    size_t numCells = gridSize.x * gridSize.y * gridSize.z;
    cellData = FixedSizeDeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);
    
    cubOutputData = FixedSizeDeviceArray<char>(sizeof(double), 1);
    cubTemporaryStorage = FixedSizeDeviceArray<char>(numBubbles * sizeof(double), 1);
    
    hostData.resize(bubbleData.getSize(), 0);
    
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
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    generateBubbles();
    deleteSmallBubbles();
    updateCellsAndNeighbors();

    // Calculate some initial values which are needed
    // for the two-step Adams-Bashforth-Moulton perdictor-corrector method (ABMpc).

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    
    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT);
    
    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_OLD);
    
    double *energies = bubbleData.getRowPtr((size_t)BubbleProperty::ENERGY);
    double *freeArea = bubbleData.getRowPtr((size_t)BubbleProperty::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    const size_t numBlocksAcc = (size_t)std::ceil(hostNumPairs / (float)numThreads);

    double timeStep = env->getTimeStep();
    
    size_t numBytesToReset = sizeof(double) * 6 * bubbleData.getWidth();
    CUDA_CALL(cudaMemset(static_cast<void*>(energies), 0, numBytesToReset));

    std::cout << "Calculating some initial values as a part of setup." << std::endl;

    calculateVelocityAndGasExchange<<<numBlocksAcc, numThreads>>>(x, y, z, r,
								  dxdtOld, dydtOld, dzdtOld, drdtOld,
								  energies,
								  freeArea,
								  firstIndices,
								  secondIndices,
								  numBubbles,
								  hostNumPairs,
								  env->getFZeroPerMuZero(),
								  env->getPi(),
								  tfr - lbb,
								  false,
								  false);
    
    eulerIntegration<<<numBlocks, numThreads>>>(x, y, z, r,
						dxdtOld, dydtOld, dzdtOld, drdtOld,
						tfr, lbb, timeStep, numBubbles);

    if (deleteSmallBubbles())
	updateCellsAndNeighbors();

    CUDA_CALL(cudaMemset(static_cast<void*>(energies), 0, numBytesToReset));
    
    calculateVelocityAndGasExchange<<<numBlocksAcc, numThreads>>>(x, y, z, r,
								  dxdtOld, dydtOld, dzdtOld, drdtOld,
								  energies,
								  freeArea,
								  firstIndices,
								  secondIndices,
								  numBubbles,
								  hostNumPairs,
								  env->getFZeroPerMuZero(),
								  env->getPi(),
								  env->getTfr() - env->getLbb(),
								  false,
								  false);
    
    NVTX_RANGE_POP();
}

bool cubble::Simulator::integrate(bool useGasExchange, bool calculateEnergy)
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    const size_t numBlocksAcc = (size_t)std::ceil(hostNumPairs / (float)numThreads);

    double timeStep = env->getTimeStep();
    double error = 0;

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    
    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Z_PRD);
    double *rPrd = bubbleData.getRowPtr((size_t)BubbleProperty::R_PRD);
    
    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dydt = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT);
    double *dzdt = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT);
    double *drdt = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT);
    
    double *dxdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_PRD);
    double *dydtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_PRD);
    double *dzdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_PRD);
    double *drdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_PRD);
    
    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
    double *dydtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DYDT_OLD);
    double *dzdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DZDT_OLD);
    double *drdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DRDT_OLD);

    double *energies = bubbleData.getRowPtr((size_t)BubbleProperty::ENERGY);
    double *errors = bubbleData.getRowPtr((size_t)BubbleProperty::ERROR);
    double *volumes = bubbleData.getRowPtr((size_t)BubbleProperty::VOLUME);
    double *freeArea = bubbleData.getRowPtr((size_t)BubbleProperty::FREE_AREA);

    int *firstIndices = neighborPairIndices.getRowPtr(0);
    int *secondIndices = neighborPairIndices.getRowPtr(1);

    do
    {
	predict<<<numBlocks, numThreads>>>(x, y, z, r,
					   xPrd, yPrd, zPrd, rPrd,
					   dxdt, dydt, dzdt, drdt,
					   dxdtOld, dydtOld, dzdtOld, drdtOld,
					   tfr, lbb, timeStep, numBubbles, useGasExchange);

	// Using atomicAdd, so these need to be reset to 0 every time before use.
	size_t numBytesToReset = sizeof(double) * 6 * bubbleData.getWidth();
	CUDA_CALL(cudaMemset(static_cast<void*>(dxdtPrd), 0, numBytesToReset));

	calculateVelocityAndGasExchange<<<numBlocksAcc, numThreads>>>(xPrd, yPrd, zPrd, rPrd,
								      dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
								      energies,
								      freeArea,
								      firstIndices,
								      secondIndices,
								      numBubbles,
								      hostNumPairs,
								      env->getFZeroPerMuZero(),
								      env->getPi(),
								      env->getTfr() - env->getLbb(),
								      calculateEnergy,
								      useGasExchange);

	if (useGasExchange)
	{
	    calculateFreeAreaPerRadius<<<numBlocks, numThreads>>>(rPrd, freeArea, errors, env->getPi(), numBubbles);
	    double invRho = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, errors, numBubbles);
	    invRho /= cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, freeArea, numBubbles);
	    calculateFinalRadiusChangeRate<<<numBlocks, numThreads>>>(drdtPrd,
								      rPrd,
								      freeArea,
								      numBubbles,
								      invRho,
								      1.0 / env->getPi(),
								      env->getKappa(),
								      env->getKParameter());
	}
        
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
        
        error = cubReduction<double, double*, double*>(&cub::DeviceReduce::Max, errors, numBubbles);

	if (error < env->getErrorTolerance() / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > env->getErrorTolerance())
	    timeStep *= 0.5;
    }
    while (error > env->getErrorTolerance());

    updateData();
    
    ++integrationStep;
    env->setTimeStep(timeStep);
    SimulationTime += timeStep;

    if (calculateEnergy)
	ElasticEnergy = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, energies, numBubbles);
    
    if (deleteSmallBubbles() || integrationStep % 100)
	updateCellsAndNeighbors();

    NVTX_RANGE_POP();

    return numBubbles > env->getMinNumBubbles();
}

void cubble::Simulator::generateBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    std::cout << "Starting to generate data for bubbles." << std::endl;
    
    const int rngSeed = env->getRngSeed();
    const double avgRad = env->getAvgRad();
    const double stdDevRad = env->getStdDevRad();
    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    
    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *yPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Y_PRD);
    double *zPrd = bubbleData.getRowPtr((size_t)BubbleProperty::Z_PRD);
    
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double *w = bubbleData.getRowPtr((size_t)BubbleProperty::R_PRD);
    
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    
    CURAND_CALL(curandGenerateUniformDouble(generator, x, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, y, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, z, numBubbles));
    CURAND_CALL(curandGenerateUniformDouble(generator, w, numBubbles));
    CURAND_CALL(curandGenerateNormalDouble(generator, r, numBubbles, avgRad, stdDevRad));

    CURAND_CALL(curandDestroyGenerator(generator));

    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil((float)numBubbles / (float)numThreads);
    assignDataToBubbles<<<numBlocks, numThreads>>>(x, y, z,
						   xPrd, yPrd, zPrd,
						   r, w, givenNumBubblesPerDim, tfr, lbb, avgRad, numBubbles);
    NVTX_RANGE_POP();
}

void cubble::Simulator::updateCellsAndNeighbors()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    //std::cout << "Updating cells and neighbors!" << std::endl;
    
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec domainDim(gridSize.x, gridSize.y, gridSize.z);
    const size_t numThreads = 128;
    size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);

    cellData.setBytesToZero();
    indicesPerCell.setBytesToZero();
    neighborPairIndices.setBytesToZero();
    numPairs.setBytesToZero();
    
    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

    calculateOffsets<<<numBlocks, numThreads>>>(x, y, z, sizes, domainDim, numBubbles, numCells);

    cubScan<int*, int*>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
    CUDA_CALL(cudaMemset(static_cast<void*>(sizes), 0, sizeof(int) * numCells));

    bubblesToCells<<<numBlocks, numThreads>>>(x, y, z,
					      indicesPerCell.getDataPtr(),
					      offsets,
					      sizes,
					      domainDim,
					      numBubbles);
    
    gridSize.z *= CUBBLE_NUM_NEIGHBORS + 1;
    assertGridSizeBelowLimit(gridSize);

    int sharedMemSizeInBytes = cubReduction<int, int*, int*>(&cub::DeviceReduce::Max, sizes, numCells);

    sharedMemSizeInBytes *= sharedMemSizeInBytes;
    sharedMemSizeInBytes *= 2;
    sharedMemSizeInBytes *= sizeof(int);

    assertMemBelowLimit(sharedMemSizeInBytes);
    assert(sharedMemSizeInBytes > 0 && "Zero bytes of shared memory reserved!");

    findBubblePairs<<<gridSize, numThreads, sharedMemSizeInBytes>>>(x, y, z, r,
								    indicesPerCell.getDataPtr(),
								    offsets,
								    sizes,
								    neighborPairIndices.getRowPtr(2),
								    neighborPairIndices.getRowPtr(3),
								    numPairs.getDataPtr(),
								    numCells,
								    numBubbles,
								    env->getTfr() - env->getLbb(),
								    sharedMemSizeInBytes / sizeof(int),
								    neighborPairIndices.getWidth());

    CUDA_CALL(cudaMemcpy(&hostNumPairs, static_cast<void*>(numPairs.getDataPtr()), sizeof(int), cudaMemcpyDeviceToHost));

    cubSortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
			   const_cast<const int*>(neighborPairIndices.getRowPtr(2)),
			   neighborPairIndices.getRowPtr(0),
			   const_cast<const int*>(neighborPairIndices.getRowPtr(3)),
			   neighborPairIndices.getRowPtr(1),
			   hostNumPairs);
    
    NVTX_RANGE_POP();
}

void cubble::Simulator::updateData()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    // x, y, z, r are in memory continuously, so we can just make three copies with 4x the data of one component.
    size_t numBytesToCopy = 4 * sizeof(double) * bubbleData.getWidth();

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *xPrd = bubbleData.getRowPtr((size_t)BubbleProperty::X_PRD);
    double *dxdt = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT);
    double *dxdtPrd = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_PRD);
    double *dxdtOld = bubbleData.getRowPtr((size_t)BubbleProperty::DXDT_OLD);
    
    CUDA_CALL(cudaMemcpyAsync(x, xPrd, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpyAsync(dxdtOld, dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpyAsync(dxdt, dxdtPrd, numBytesToCopy, cudaMemcpyDeviceToDevice));

    NVTX_RANGE_POP();
}

bool cubble::Simulator::deleteSmallBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double minRadius = cubReduction<double, double*, double*>(&cub::DeviceReduce::Min, r, numBubbles);

    bool atLeastOneBubbleDeleted = false;
    
    if (minRadius < env->getMinRad())
    {
	NVTX_RANGE_PUSH_A("BubbleRemoval");
	
	CUDA_CALL(cudaMemcpyAsync(hostData.data(),
				  bubbleData.getDataPtr(),
				  bubbleData.getSizeInBytes(),
				  cudaMemcpyDeviceToHost));
	
	minRadius = env->getMinRad();
	size_t memoryStride = bubbleData.getWidth();
	size_t rIdx = (size_t)(size_t)BubbleProperty::R;
	std::vector<int> idxVec;
	
	for (size_t i = 0; i < (size_t)BubbleProperty::NUM_VALUES; ++i)
	    idxVec.push_back(i);

	// Synchronize if memcpy not done yet.
	CUDA_CALL(cudaDeviceSynchronize());

	double volumeMultiplier = 0;
	for (int i = (int)numBubbles - 1; i > -1; --i)
	{
	    double radius = hostData[rIdx * memoryStride + i];
	    assert(radius > 0 && "Radius is negative!");
	    if (radius < minRadius)
	    {
		double volume = 0;
	        volume = radius * radius;
#if (NUM_DIM == 3)
		volume *= 1.333333333333333333333333 * radius;
#endif
		volume *= env->getPi();
	        volumeMultiplier += volume;
		
		for (size_t j = 0; j < idxVec.size(); ++j)
		    hostData[j * memoryStride + i] = hostData[j * memoryStride + (numBubbles - 1)];
		--numBubbles;
	    }
	}

	CUDA_CALL(cudaMemcpy(bubbleData.getDataPtr(),
			     hostData.data(),
			     bubbleData.getSizeInBytes(),
			     cudaMemcpyHostToDevice));

        volumeMultiplier /= getVolumeOfBubbles();
	volumeMultiplier += 1.0;

#if (NUM_DIM == 3)
	volumeMultiplier = std::cbrt(volumeMultiplier);
#else
	volumeMultiplier = std::sqrt(volumeMultiplier);
#endif

	addVolume<<<numBlocks, numThreads>>>(r, numBubbles, volumeMultiplier);
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	NVTX_RANGE_POP();
	
	atLeastOneBubbleDeleted = true;
    }
    
    NVTX_RANGE_POP();

    return atLeastOneBubbleDeleted;
}

dim3 cubble::Simulator::getGridSize()
{
    int numBubblesPerCell = env->getNumBubblesPerCell();
#if (NUM_DIM == 3)
    int numCellsPerDim = std::ceil(std::cbrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, numCellsPerDim);
#else
    int numCellsPerDim = std::ceil(std::sqrt((float)numBubbles / numBubblesPerCell));
    dim3 gridSize(numCellsPerDim, numCellsPerDim, 1);
#endif
    
    return gridSize;
}

double cubble::Simulator::getVolumeOfBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);

    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);

    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double *volPtr = bubbleData.getRowPtr((size_t)BubbleProperty::VOLUME);
    calculateVolumes<<<numBlocks, numThreads>>>(r, volPtr, numBubbles, env->getPi());
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    double volume = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, volPtr, numBubbles);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    NVTX_RANGE_POP();
    
    return volume;
}

double cubble::Simulator::getAverageRadius()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double avgRad = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, r, numBubbles);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    avgRad/= numBubbles;
    
    NVTX_RANGE_POP();
    
    return avgRad;
}

void cubble::Simulator::getBubbles(std::vector<Bubble> &bubbles) const
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    bubbles.clear();
    bubbles.resize(numBubbles);

    size_t memoryStride = bubbleData.getWidth();
    double *devX = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    std::vector<double> xyzr;
    xyzr.resize(memoryStride * 4);

    CUDA_CALL(cudaMemcpy(xyzr.data(), devX, sizeof(double) * 4 * memoryStride, cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < numBubbles; ++i)
    {
	Bubble b;
	dvec pos(-1, -1, -1);
	pos.x = xyzr[i];
	pos.y = xyzr[i + memoryStride];
	pos.z = xyzr[i + 2 * memoryStride];
	b.setPos(pos);
	b.setRadius(xyzr[i + 3 * memoryStride]);
	bubbles[i] = b;
    }
    
    NVTX_RANGE_POP();
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

	double radius = r[tid];
	r[tid] = radius > 0 ? radius : -radius;
	w[tid] = r[tid];
    }
}

__global__
void cubble::calculateOffsets(double *x,
			      double *y,
			      double *z,
			      int *sizes,
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
	
	atomicAdd(&sizes[index], 1);
    }
}

__global__
void cubble::bubblesToCells(double *x,
			    double *y,
			    double *z,
			    int *indices,
			    int *offsets,
			    int *sizes,
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
	const int offset = offsets[index] + atomicAdd(&sizes[index], 1);
        indices[offset] = tid;
    }
}

__global__
void cubble::findBubblePairs(double *x,
			     double *y,
			     double *z,
			     double *r,
			     int *indices,
			     int *offsets,
			     int *sizes,
			     int *firstIndices,
			     int *secondIndices,
			     int *numPairs,
			     int numCells,
			     int numBubbles,
			     dvec interval,
			     int maxNumSharedVals,
			     int maxNumPairs)
{
    __shared__ int numLocalPairs[1];
    extern __shared__ int localPairs[];

    DEVICE_ASSERT(numCells > 0);
    DEVICE_ASSERT(numBubbles > 0);

    if (threadIdx.x == 0)
	numLocalPairs[0] = 0;
    
    __syncthreads();
    
#if (NUM_DIM == 3)
    const int numNeighborCells = 14;
#else
    const int numNeighborCells = 5;
#endif

    const int selfCellIndex = blockIdx.z / numNeighborCells * gridDim.y * gridDim.x
	+ blockIdx.y * gridDim.x
	+ blockIdx.x;
    const int neighborCellIndex = getNeighborCellIndex(ivec(blockIdx.x, blockIdx.y, blockIdx.z / numNeighborCells),
						       ivec(gridDim.x, gridDim.y, gridDim.z / numNeighborCells),
						       blockIdx.z % numNeighborCells);
    DEVICE_ASSERT(neighborCellIndex < numCells);
    DEVICE_ASSERT(selfCellIndex < numCells);

    const bool selfComparison = selfCellIndex == neighborCellIndex;
    const int selfSize = sizes[selfCellIndex];
    const int selfOffset = offsets[selfCellIndex];
    const int neighborSize = sizes[neighborCellIndex];
    const int neighborOffset = offsets[neighborCellIndex];
    int numComparisons = selfSize * neighborSize;

    DEVICE_ASSERT(selfOffset < numBubbles);
    DEVICE_ASSERT(neighborOffset < numBubbles);
    DEVICE_ASSERT(neighborSize < numBubbles);
    DEVICE_ASSERT(selfSize < numBubbles);

    int id = 0;
    for (int i = 0; i < (1 + numComparisons / blockDim.x); ++i)
    {
        id = i * blockDim.x + threadIdx.x;
	if (id < numComparisons)
	{
	    int idx1 = id / neighborSize;
	    int idx2 = id % neighborSize;

	    DEVICE_ASSERT(selfOffset + idx1 < numBubbles);
	    DEVICE_ASSERT(neighborOffset + idx2 < numBubbles);

	    idx1 = indices[selfOffset + idx1];
	    idx2 = indices[neighborOffset + idx2];

	    if (idx1 == idx2 || (selfComparison && idx2 < idx1))
		continue;
	    
	    DEVICE_ASSERT(idx1 < numBubbles);
	    DEVICE_ASSERT(idx2 < numBubbles);

	    double wrappedComponent = getWrappedCoordinate(x[idx1], x[idx2], interval.x);
	    double magnitude = wrappedComponent * wrappedComponent;
	    
	    wrappedComponent = getWrappedCoordinate(y[idx1], y[idx2], interval.y);
	    magnitude += wrappedComponent * wrappedComponent;
	    
	    wrappedComponent = getWrappedCoordinate(z[idx1], z[idx2], interval.z);
	    magnitude += wrappedComponent * wrappedComponent;
	    
	    wrappedComponent = r[idx1] + r[idx2];
	    wrappedComponent *= wrappedComponent;

	    if (magnitude < wrappedComponent)
	    {
		// Set the smaller index to idx1 and larger to idx2
		id = idx1;
		idx1 = idx1 > idx2 ? idx2 : idx1;
		idx2 = idx1 == idx2 ? id : idx2;
		    
		id = atomicAdd(numLocalPairs, 2);
		DEVICE_ASSERT(id < numComparisons * 2);
		DEVICE_ASSERT(id + 1 < maxNumSharedVals);
		localPairs[id] = idx1;
		localPairs[id + 1] = idx2;
	    }
	}
    }

    __syncthreads();

    numComparisons = numLocalPairs[0] / 2;

    __syncthreads();

    if (threadIdx.x == 0)
	numLocalPairs[0] = atomicAdd(numPairs, numComparisons);

    __syncthreads();
    
    for (int i = 0; i < (1 + numComparisons / blockDim.x); ++i)
    {
	id = i * blockDim.x + threadIdx.x;
	if (id < numComparisons)
	{
	    DEVICE_ASSERT(2 * id + 1 < maxNumSharedVals);
	    DEVICE_ASSERT(numLocalPairs[0] + id < maxNumPairs);
	    firstIndices[numLocalPairs[0] + id] = localPairs[2 * id];
	    secondIndices[numLocalPairs[0] + id] = localPairs[2 * id + 1];
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
void cubble::calculateVelocityAndGasExchange(double *x,
					     double *y,
					     double *z,
					     double *r,
					     
					     double *dxdt,
					     double *dydt,
					     double *dzdt,
					     double *drdt,

					     double *energy,
					     double *freeArea,
					     
					     int *firstIndices,
					     int *secondIndices,
					     
					     int numBubbles,
					     int numPairs,
					     double fZeroPerMuZero,
					     double pi,
					     dvec interval,
					     bool calculateEnergy,
					     bool useGasExchange)
{
    // FYI: This kernel heavily reuses variables, since kernels can easily become register bound.
    // Pay attention to the last assignation of a variable.
    
    const int tid = getGlobalTid();
    if (tid < numPairs)
    {
	const int idx1 = firstIndices[tid];
	const int idx2 = secondIndices[tid];

	DEVICE_ASSERT(idx1 < numBubbles);
	DEVICE_ASSERT(idx2 < numBubbles);
	DEVICE_ASSERT(idx1 != idx2);

	double velX = getWrappedCoordinate(x[idx1], x[idx2], interval.x);
	double magnitude = velX * velX;
	
	double velY = getWrappedCoordinate(y[idx1], y[idx2], interval.y);
        magnitude += velY * velY;

	double velZ = 0;
#if (NUM_DIM == 3)
        velZ = getWrappedCoordinate(z[idx1], z[idx2], interval.z);
        magnitude += velZ * velZ;
#endif

	DEVICE_ASSERT(magnitude > 0);
	magnitude = sqrt(magnitude);

	double generalVariable = r[idx1] + r[idx2];
	DEVICE_ASSERT(generalVariable > 0);
	double invRadii = 1.0 / generalVariable;

	if (calculateEnergy)
	{
	    generalVariable = generalVariable - magnitude;
	    generalVariable *= generalVariable;
	    generalVariable *= invRadii;

	    atomicAdd(&energy[idx1], generalVariable);
	    atomicAdd(&energy[idx2], generalVariable);
	}

        double invMagnitude = 1.0 / magnitude;
        generalVariable = fZeroPerMuZero * (invMagnitude - invRadii);
	
        velX *= generalVariable;
	velY *= generalVariable;
	velZ *= generalVariable;

	atomicAdd(&dxdt[idx1], velX);
	atomicAdd(&dxdt[idx2], -velX);
	
	atomicAdd(&dydt[idx1], velY);
	atomicAdd(&dydt[idx2], -velY);
#if (NUM_DIM == 3)
	atomicAdd(&dzdt[idx1], velZ);
	atomicAdd(&dzdt[idx2], -velZ);
#endif

	if (useGasExchange)
	{
	    velX = r[idx1];
	    velY = r[idx2];
	    DEVICE_ASSERT(magnitude > velX && magnitude > velY);
	    generalVariable = velY * velY;
	    velZ = 0.5 * (generalVariable - velX * velX + magnitude * magnitude) * invMagnitude;
	    velZ *= velZ;
	    velZ = generalVariable - velZ;
	    DEVICE_ASSERT(velZ > -0.0001);
	    velZ = velZ < 0 ? -velZ : velZ;
	    DEVICE_ASSERT(velZ >= 0);
	    
#if (NUM_DIM == 3)
	    velZ *= pi;
#else
	    velZ = 2.0 * sqrt(velZ);
#endif
	    atomicAdd(&freeArea[idx1], velZ);
	    atomicAdd(&freeArea[idx2], velZ);
	    
	    velZ *= 1.0 / velY - 1.0 / velX;
	    
	    atomicAdd(&drdt[idx1], velZ);
	    atomicAdd(&drdt[idx2], -velZ);
	}
    }
}

__global__
void cubble::calculateFreeAreaPerRadius(double *r, double *freeArea, double *output, double pi, int numBubbles)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double area = 2.0 * pi * r[tid];
#if (NUM_DIM == 3)
	area *= 2.0 * r[tid];
#endif
	area -= freeArea[tid];
	freeArea[tid] = area;
	output[tid] = freeArea[tid] / r[tid];
    }
}

__global__
void cubble::calculateFinalRadiusChangeRate(double *drdt,
					    double *r,
					    double *freeArea,
					    int numBubbles,
					    double invRho,
					    double invPi,
					    double kappa,
					    double kParam)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double vr = kappa * freeArea[tid] * (invRho - 1.0 / r[tid]);
	vr += drdt[tid];
	
#if (NUM_DIM == 3)
	vr *= 0.25 * invPi;
	vr /= r[tid] * r[tid];
#else
	vr *= 0.5 * invPi;
	vr /= r[tid];
#endif
	
	drdt[tid] = kParam * vr;
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
void cubble::addVolume(double *r, int numBubbles, double volumeMultiplier)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
	r[tid] = r[tid] * volumeMultiplier;
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
double cubble::getWrappedCoordinate(double val1, double val2, double multiplier)
{
    DEVICE_ASSERT(val1 <= 1.0 && val2 <= 1.0);
    DEVICE_ASSERT(val1 >= 0.0 && val2 >= 0.0);
    double difference = val1 - val2;
    val2 = difference < -0.5 ? val2 - 1.0 : (difference > 0.5 ? val2 + 1.0 : val2);
    val2 = val1 - val2;
    
    return val2 * multiplier;
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
