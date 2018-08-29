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
    bubblePairData = FixedSizeDeviceArray<double>(numBubbles,
						  neighborStride,
						  (size_t)BubblePairProperty::NUM_VALUES);
    miscData = FixedSizeDeviceArray<int>(numBubbles, (size_t)MiscIntProperty::NUM_VALUES);
    neighborIndices = FixedSizeDeviceArray<int>(numBubbles, neighborStride);
    
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
    assignBubblesToCells();

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

    double *ax = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_X);
    double *ay = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_Y);
    double *az = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_Z);
    double *ar = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_R);
    double *e = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ENERGY);
    double *areaOverlap = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::OVERLAP_AREA);

    int *numNeighbors = miscData.getRowPtr((size_t)MiscIntProperty::NUM_NEIGHBORS);

    const dvec tfr = env->getTfr();
    const dvec lbb = env->getLbb();
    const double minRad = env->getMinRad();
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    const size_t numBlocksForAcc = (size_t)std::ceil(numBubbles * neighborStride / (float)numThreads);

    double timeStep = env->getTimeStep();

    createAccelerationArray<<<numBlocksForAcc, numThreads>>>(x, y, z, r,
							     ax, ay, az, ar, e, areaOverlap,
							     numNeighbors,
							     neighborIndices.getDataPtr(),
							     tfr - lbb,
							     numBubbles,
							     neighborStride,
							     env->getPi(),
							     false,
							     false);
    
    calculateVelocityFromAccelerations<<<numBlocks, numThreads>>>(ax, ay, az, ar, e, areaOverlap,
								  dxdtOld, dydtOld, dzdtOld, drdtOld,
								  freeArea,
								  energies,
								  numBubbles,
								  neighborStride,
								  env->getFZeroPerMuZero(),
								  false,
								  false);
    
    eulerIntegration<<<numBlocks, numThreads>>>(x, y, z, r,
						dxdtOld, dydtOld, dzdtOld, drdtOld,
						tfr, lbb, timeStep, numBubbles);
    
    createAccelerationArray<<<numBlocksForAcc, numThreads>>>(x, y, z, r,
							     ax, ay, az, ar, e, areaOverlap,
							     numNeighbors,
							     neighborIndices.getDataPtr(),
							     tfr - lbb,
							     numBubbles,
							     neighborStride,
							     env->getPi(),
							     false,
							     false);
    
    calculateVelocityFromAccelerations<<<numBlocks, numThreads>>>(ax, ay, az, ar, e, areaOverlap,
								  dxdtOld, dydtOld, dzdtOld, drdtOld,
								  freeArea,
								  energies,
								  numBubbles,
								  neighborStride,
								  env->getFZeroPerMuZero(),
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
    const size_t numBlocksForAcc = (size_t)std::ceil(numBubbles * neighborStride / (float)numThreads);

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

    double *ax = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_X);
    double *ay = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_Y);
    double *az = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_Z);
    double *ar = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ACCELERATION_R);
    double *e = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::ENERGY);
    double *areaOverlap = bubblePairData.getRowPtr(0, (size_t)BubblePairProperty::OVERLAP_AREA);

    int *numNeighbors = miscData.getRowPtr((size_t)MiscIntProperty::NUM_NEIGHBORS);
    
    do
    {
	predict<<<numBlocks, numThreads>>>(x, y, z, r,
					   xPrd, yPrd, zPrd, rPrd,
					   dxdt, dydt, dzdt, drdt,
					   dxdtOld, dydtOld, dzdtOld, drdtOld,
					   tfr, lbb, timeStep, numBubbles, useGasExchange);
	
	createAccelerationArray<<<numBlocksForAcc, numThreads>>>(xPrd, yPrd, zPrd, rPrd,
								 ax, ay, az, ar, e, areaOverlap,
								 numNeighbors,
								 neighborIndices.getDataPtr(),
								 tfr - lbb,
								 numBubbles,
								 neighborStride,
								 env->getPi(),
								 useGasExchange,
								 calculateEnergy);
	
	calculateVelocityFromAccelerations<<<numBlocks, numThreads>>>(ax, ay, az, ar, e, areaOverlap,
								      dxdtPrd, dydtPrd, dzdtPrd, drdtPrd,
								      freeArea,
								      energies,
								      numBubbles,
								      neighborStride,
								      env->getFZeroPerMuZero(),
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
    
    deleteSmallBubbles();

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
    
    std::cout << "\tGenerating data..." << std::endl;

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

    std::cout << "\tAssigning data to bubbles..." << std::endl;;

    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil((float)numBubbles / (float)numThreads);
    assignDataToBubbles<<<numBlocks, numThreads>>>(x, y, z,
						   xPrd, yPrd, zPrd,
						   r, w, givenNumBubblesPerDim, tfr, lbb, avgRad, numBubbles);
    NVTX_RANGE_POP();
}

void cubble::Simulator::assignBubblesToCells()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    dim3 gridSize = getGridSize();
    const int numCells = gridSize.x * gridSize.y * gridSize.z;
    const dvec domainDim(gridSize.x, gridSize.y, gridSize.z);
    const dvec cellSize = (env->getTfr() - env->getLbb()) / domainDim;
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    const int numDomains = (CUBBLE_NUM_NEIGHBORS + 1) * 4;

    double *x = bubbleData.getRowPtr((size_t)BubbleProperty::X);
    double *y = bubbleData.getRowPtr((size_t)BubbleProperty::Y);
    double *z = bubbleData.getRowPtr((size_t)BubbleProperty::Z);
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);

    int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
    int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);
    int *numNeighbors = miscData.getRowPtr((size_t)MiscIntProperty::NUM_NEIGHBORS);
    int *indices = miscData.getRowPtr((size_t)MiscIntProperty::INDEX);

    cellData.setBytesToZero();
    miscData.setBytesToZero();
    neighborIndices.setBytesToZero();
    
    calculateOffsets<<<numBlocks, numThreads>>>(x, y, z, sizes, domainDim, numBubbles, numCells);

    cubScan<int*, int*>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
    cudaMemset(static_cast<void*>(sizes), 0, sizeof(int) * numCells);

    bubblesToCells<<<numBlocks, numThreads>>>(x, y, z, indices, offsets, sizes, domainDim, numBubbles);

    gridSize.z *= numDomains;
    assertGridSizeBelowLimit(gridSize);
    
    findNeighbors<<<gridSize, numThreads>>>(x, y, z, r,
					    indices,
					    offsets,
					    sizes,
					    numNeighbors,
					    neighborIndices.getDataPtr(),
					    env->getTfr(),
					    env->getLbb(),
					    numBubbles,
					    numDomains,
					    numCells,
					    neighborStride);
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
    
    cudaMemcpyAsync(x, xPrd, numBytesToCopy, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dxdtOld, dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dxdt, dxdtPrd, numBytesToCopy, cudaMemcpyDeviceToDevice);

    NVTX_RANGE_POP();
}

void cubble::Simulator::deleteSmallBubbles()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    const size_t numThreads = 128;
    const size_t numBlocks = (size_t)std::ceil(numBubbles / (float)numThreads);
    
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double minRadius = cubReduction<double, double*, double*>(&cub::DeviceReduce::Min, r, numBubbles);
    
    if (minRadius < env->getMinRad())
    {
	NVTX_RANGE_PUSH_A("BubbleRemoval");
	
	cudaMemcpyAsync(hostData.data(),
		        bubbleData.getDataPtr(),
		        bubbleData.getSizeInBytes(),
			cudaMemcpyDeviceToHost);
	
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

	cudaMemcpyAsync(bubbleData.getDataPtr(),
			hostData.data(),
		        bubbleData.getSizeInBytes(),
			cudaMemcpyHostToDevice);

        volumeMultiplier /= getVolumeOfBubbles();
	volumeMultiplier += 1.0;

#if (NUM_DIM == 3)
	volumeMultiplier = std::cbrt(volumeMultiplier);
#else
	volumeMultiplier = std::sqrt(volumeMultiplier);
#endif

	addVolume<<<numBlocks, numThreads>>>(r, numBubbles, volumeMultiplier);
	
	NVTX_RANGE_POP();
	
	assignBubblesToCells();
    }
    else if (integrationStep % 100)
	assignBubblesToCells();

    NVTX_RANGE_POP();
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
    double volume = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, volPtr, numBubbles);
    
    NVTX_RANGE_POP();
    
    return volume;
}

double cubble::Simulator::getAverageRadius()
{
    NVTX_RANGE_PUSH_A(__FUNCTION__);
    
    double *r = bubbleData.getRowPtr((size_t)BubbleProperty::R);
    double avgRad = cubReduction<double, double*, double*>(&cub::DeviceReduce::Sum, r, numBubbles);
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

    cudaMemcpy(xyzr.data(), devX, sizeof(double) * 4 * memoryStride, cudaMemcpyDeviceToHost);
    
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
void cubble::findNeighbors(double *x,
			   double *y,
			   double *z,
			   double *r,
			   int *indices,
			   int *offsets,
			   int *sizes,
			   int *numNeighbors,
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
				 offsets,
				 sizes,
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
		int index = atomicAdd(&numNeighbors[gid1], 1);
		DEVICE_ASSERT(index < neighborStride);
		index = numBubbles * index + gid1;
		DEVICE_ASSERT(index < numBubbles * neighborStride);
		neighborIndices[index] = gid2;

		if (!isOwnCell)
		{
		    index = atomicAdd(&numNeighbors[gid2], 1);
		    DEVICE_ASSERT(index < neighborStride);
		    index = numBubbles * index + gid2;
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
void cubble::createAccelerationArray(double *x,
				     double *y,
				     double *z,
				     double *r,

				     double *ax,
				     double *ay,
				     double *az,
				     double *ar,
				     double *e,
				     double *areaOverlap,
				     
				     int *numberOfNeighbors,
				     int *neighborIndices,
				     dvec interval,
				     int numBubbles,
				     int neighborStride,
				     double pi,
				     bool useGasExchange,
				     bool calculateEnergy)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles * neighborStride)
    {
	const int idx1 = tid % numBubbles;
	const int neighborNum = tid / numBubbles;

	// Accelerations recide in temporary memory which is also used by reductions
	// and other temporary things, so they might contain garbage data.
	// Later these arrays are summed, so it's important to zero all the values.
	ax[tid] = 0;
	ay[tid] = 0;
	az[tid] = 0;
	ar[tid] = 0;
	areaOverlap[tid] = 0;
	e[tid] = 0;

	if (neighborNum < numberOfNeighbors[idx1])
	{
	    const int idx2 = neighborIndices[tid];
	    
	    double x1 = x[idx1];
	    double y1 = y[idx1];
	    double z1 = z[idx1];
	    double r1 = r[idx1];
	    
	    double x2 = x[idx2];
	    double y2 = y[idx2];
	    double z2 = z[idx2];
	    double r2 = r[idx2];
	    
	    double radii = r1 + r2;
	    
	    double magnitude = x1 - x2;
	    x2 = magnitude < -0.5 ? x2 - 1.0 : (magnitude > 0.5 ? x2 + 1.0 : x2);
	    x2 = x1 - x2;
	    x2 *= interval.x;
	    
	    magnitude = y1 - y2;
	    y2 = magnitude < -0.5 ? y2 - 1.0 : (magnitude > 0.5 ? y2 + 1.0 : y2);
	    y2 = y1 - y2;
	    y2 *= interval.y;
	    
	    magnitude = z1 - z2;
	    z2 = magnitude < -0.5 ? z2 - 1.0 : (magnitude > 0.5 ? z2 + 1.0 : z2);
	    z2 = z1 - z2;
	    z2 *= interval.z;
	    
	    magnitude = sqrt(x2 * x2 + y2 * y2 + z2 * z2);
	    DEVICE_ASSERT(magnitude > 0);
	    DEVICE_ASSERT(radii > 0);
	    
	    double tempVal = 0;
	    double invRadii = 1.0 / radii;
	    if (calculateEnergy)
	    {
	        tempVal = radii - magnitude;
		tempVal *= tempVal;
		tempVal *= invRadii;
		
		e[tid] = tempVal;
	    }
	    
	    tempVal = 1.0 / magnitude;
	    
	    x2 *= tempVal - invRadii;
	    y2 *= tempVal - invRadii;
	    z2 *= tempVal - invRadii;

	    ax[tid] = x2;
	    ay[tid] = y2;
	    az[tid] = z2;
	    
	    if (useGasExchange)
	    {
		if (magnitude > r1 && magnitude > r2)
		{
		    radii = r2 * r2;
		    tempVal = 0.5 * (radii - r1 * r1 + magnitude * magnitude) * tempVal;
		    tempVal *= tempVal;
		    tempVal = radii - tempVal;
		    DEVICE_ASSERT(tempVal > -0.001);
		    tempVal = tempVal < 0 ? -tempVal : tempVal;
		    DEVICE_ASSERT(tempVal >= 0);
		    
#if (NUM_DIM == 3)
		    tempVal = tempVal * 0.25;
		    tempVal /= r1 * r1;
#else
		    tempVal = sqrt(tempVal) / (pi * r1);
#endif
		    areaOverlap[tid] = tempVal;
		    
		    tempVal *= 1.0 / r2 - 1.0 / r1;
		}
		else
		    tempVal = 0.0;
		
		ar[tid] = tempVal;
	    }
	}
    }
}

__global__
void cubble::calculateVelocityFromAccelerations(double *ax,
						double *ay,
						double *az,
						double *ar,
						double *e,
						double *areaOverlap,
			
						double *dxdt,
						double *dydt,
						double *dzdt,
						double *drdt,

						double *freeArea,
						double *energies,

						int numBubbles,
						int neighborStride,
						double fZeroPerMuZero,
						bool calculateEnergy,
						bool useGasExchange)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	double vx = 0.0;
	double vy = 0.0;
	double vz = 0.0;
	double vr = 0.0;
	double energy = 0;
	double a = 0;
	
	if (useGasExchange && calculateEnergy)
	{
	    for (int i = 0; i < neighborStride; ++i)
	    {
		vx += ax[tid + i * numBubbles];
		vy += ay[tid + i * numBubbles];
		vz += az[tid + i * numBubbles];
		vr += ar[tid + i * numBubbles];
		a += areaOverlap[tid + i * numBubbles];
		energy += e[tid + i * numBubbles];
	    }
	}
	else if (useGasExchange)
	{
	    for (int i = 0; i < neighborStride; ++i)
	    {
		vx += ax[tid + i * numBubbles];
		vy += ay[tid + i * numBubbles];
		vz += az[tid + i * numBubbles];
		vr += ar[tid + i * numBubbles];
		a += areaOverlap[tid + i * numBubbles];
	    }
	}
	else if (calculateEnergy)
	{
	    for (int i = 0; i < neighborStride; ++i)
	    {
		vx += ax[tid + i * numBubbles];
		vy += ay[tid + i * numBubbles];
		vz += az[tid + i * numBubbles];
		energy += e[tid + i * numBubbles];
	    }
	}
	else
	{
	    for (int i = 0; i < neighborStride; ++i)
	    {
		vx += ax[tid + i * numBubbles];
		vy += ay[tid + i * numBubbles];
		vz += az[tid + i * numBubbles];
	    }
	}
	    
	dxdt[tid] = vx * fZeroPerMuZero;
	dydt[tid] = vy * fZeroPerMuZero;
	dzdt[tid] = vz * fZeroPerMuZero;

	if (useGasExchange)
	{
	    drdt[tid] = vr;
	    DEVICE_ASSERT(a < 1.0);
	    freeArea[tid] = 1.0 - a;
	}
	
	if (calculateEnergy)
	    energies[tid] = energy;
    }
}

__global__
void cubble::calculateFreeAreaPerRadius(double *r, double *freeArea, double *output, double pi, int numBubbles)
{
    const int tid = getGlobalTid();
    if (tid < numBubbles)
    {
	// Seemingly redundant operations (first multiply, then divide) but freeArea is summed separately later.
#if (NUM_DIM == 3)
	freeArea[tid] *= 4.0 * pi * r[tid] * r[tid];
#else
	freeArea[tid] *= 2.0 * pi * r[tid];
#endif
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
#if (NUM_DIM == 3)
	vr *= 0.25 * invPi;
	vr /= r[tid] * r[tid];
#else
	vr *= 0.5 * invPi;
	vr /= r[tid];
#endif
	vr += drdt[tid];
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
void cubble::getDomainOffsetsAndIntervals(int numBubbles,
					  int numDomains,
					  int numCells,
					  ivec cellIdxVec,
					  ivec boxDim,
					  int *offsets,
					  int *sizes,
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
    int selfOffset = offsets[selfCellIndex];
    int selfSize = sizes[selfCellIndex];

    // Find the neighbor of this cell
    int neighborCellIndex = getNeighborCellIndex(cellIdxVec, boxDim, dj / 2);
    DEVICE_ASSERT(neighborCellIndex < numCells);
    int neighborOffset = offsets[neighborCellIndex];
    int neighborSize = sizes[neighborCellIndex];
    
    outIsOwnCell = selfCellIndex == neighborCellIndex;

    // Find the interval of values to use:
    // x-axis uses the right or the left half of the neighbor cell
    int halfSize = 0.5f * neighborSize;
    outXBegin = neighborOffset + djMod2 * halfSize;
    outXInterval = halfSize + djMod2 * (neighborSize % 2);
    
    DEVICE_ASSERT(outXBegin + outXInterval <= numBubbles);
    DEVICE_ASSERT(outXBegin + outXInterval <= neighborSize + neighborOffset);
    DEVICE_ASSERT(outXInterval == halfSize || outXInterval == halfSize + 1);

    // y-axis uses the top or bottom half of this cell
    halfSize = 0.5f * selfSize;
    outYBegin = selfOffset + di * halfSize;
    outYInterval = halfSize + di * (selfSize % 2);

    DEVICE_ASSERT(outYBegin + outYInterval <= numBubbles);
    DEVICE_ASSERT(outYInterval == halfSize || outYInterval == halfSize + 1);
    DEVICE_ASSERT(outYBegin + outYInterval <= selfSize + selfOffset);
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
