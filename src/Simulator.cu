// -*- C++ -*-

#include "Simulator.cuh"
#include "Macros.h"
#include "Vec.h"
#include "Kernels.cuh"

#include "cub/cub/cub.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <curand.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

namespace cubble
{
bool Simulator::init(const char *inputFileName, const char *outputFileName)
{
	properties = Env(inputFileName, outputFileName);
	properties.readParameters();
	timeScalingFactor = properties.getKParameter() / (properties.getAvgRad() * properties.getAvgRad());

	dvec relDim = properties.getBoxRelativeDimensions();
	relDim /= relDim.x;
	const float d = 2 * properties.getAvgRad();
#if (NUM_DIM == 3)
	const float x = std::cbrt(properties.getNumBubbles() * d * d * d / (relDim.y * relDim.z));
	dvec tfr = relDim * x;
	const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), std::ceil(tfr.z / d));
	numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
#else
	const float x = std::sqrt(properties.getNumBubbles() * d * d / relDim.y);
	dvec tfr = relDim * x;
	tfr.z = 0;
	const ivec bubblesPerDim(std::ceil(tfr.x / d), std::ceil(tfr.y / d), 0);
	numBubbles = bubblesPerDim.x * bubblesPerDim.y;
#endif
	bubblesPerDimAtStart = bubblesPerDim;
	tfr = d * bubblesPerDim.asType<double>();
	properties.setTfr(tfr + properties.getLbb());
	dvec interval = properties.getTfr() - properties.getLbb();
	properties.setFlowTfr(interval * properties.getFlowTfr() + properties.getLbb());
	properties.setFlowLbb(interval * properties.getFlowLbb() + properties.getLbb());

	cubWrapper = std::make_shared<CubWrapper>(numBubbles * sizeof(double));

	dataStride = numBubbles + !!(numBubbles % 32) * (32 - numBubbles % 32);
	CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&deviceData), sizeof(double) * dataStride * numAliases));
	adp.x = deviceData;
	adp.y = deviceData + 1 * dataStride;
	adp.z = deviceData + 2 * dataStride;
	adp.r = deviceData + 3 * dataStride;
	adp.dxdt = deviceData + 4 * dataStride;
	adp.dydt = deviceData + 5 * dataStride;
	adp.dzdt = deviceData + 6 * dataStride;
	adp.drdt = deviceData + 7 * dataStride;
	adp.dxdtO = deviceData + 8 * dataStride;
	adp.dydtO = deviceData + 9 * dataStride;
	adp.dzdtO = deviceData + 10 * dataStride;
	adp.drdtO = deviceData + 11 * dataStride;
	adp.x0 = deviceData + 12 * dataStride;
	adp.y0 = deviceData + 13 * dataStride;
	adp.z0 = deviceData + 14 * dataStride;
	adp.s = deviceData + 15 * dataStride;
	adp.d = deviceData + 16 * dataStride;
	adp.xP = deviceData + 17 * dataStride;
	adp.yP = deviceData + 18 * dataStride;
	adp.zP = deviceData + 19 * dataStride;
	adp.rP = deviceData + 20 * dataStride;
	adp.dxdtP = deviceData + 21 * dataStride;
	adp.dydtP = deviceData + 22 * dataStride;
	adp.dzdtP = deviceData + 23 * dataStride;
	adp.drdtP = deviceData + 24 * dataStride;
	adp.error = deviceData + 25 * dataStride;
	adp.dummy1 = deviceData + 26 * dataStride;
	adp.dummy2 = deviceData + 27 * dataStride;
	adp.dummy3 = deviceData + 28 * dataStride;
	adp.dummy4 = deviceData + 29 * dataStride;
	adp.dummy5 = deviceData + 30 * dataStride;
	adp.dummy6 = deviceData + 31 * dataStride;
	adp.dummy7 = deviceData + 32 * dataStride;
	adp.dummy8 = deviceData + 33 * dataStride;

	aboveMinRadFlags = DeviceArray<int>(dataStride, 2u);
	bubbleCellIndices = DeviceArray<int>(dataStride, 4u);
	pairs = DeviceArray<int>(8 * dataStride, 4u);
	wrapMultipliers = DeviceArray<int>(dataStride, 6);

	const dim3 gridSize = getGridSize();
	size_t numCells = gridSize.x * gridSize.y * gridSize.z;
	cellData = DeviceArray<int>(numCells, (size_t)CellProperty::NUM_VALUES);

	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfa), dTotalFreeArea));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtfapr), dTotalFreeAreaPerRadius));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&mbpc), dMaxBubblesPerCell));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dvm), dVolumeMultiplier));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dtv), dTotalVolume));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&np), dNumPairs));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dir), dInvRho));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dta), dTotalArea));
	CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&dasai), dAverageSurfaceAreaIn));

	CUDA_ASSERT(cudaStreamCreateWithFlags(&nonBlockingStream1, cudaStreamNonBlocking));
	CUDA_ASSERT(cudaStreamCreateWithFlags(&nonBlockingStream2, cudaStreamNonBlocking));
	CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent1, cudaEventBlockingSync));
	CUDA_ASSERT(cudaEventCreateWithFlags(&blockingEvent2, cudaEventBlockingSync));

	for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
	{
		neighborStreamVec.emplace_back();
		neighborEventVec.emplace_back();
		CUDA_ASSERT(cudaStreamCreateWithFlags(&neighborStreamVec[i], cudaStreamNonBlocking));
		CUDA_ASSERT(cudaEventCreate(&neighborEventVec[i]));
	}

	pinnedInt = PinnedHostArray<int>(1);
	pinnedDouble = PinnedHostArray<double>(3);

	printRelevantInfoOfCurrentDevice();

	pairKernelSize.block = dim3(128, 1, 1);
	pairKernelSize.grid = dim3(256, 1, 1);

	return true;
}

void Simulator::deinit()
{
	saveSnapshotToFile();
	properties.writeParameters();

	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaFree(static_cast<void *>(deviceData)));

	CUDA_CALL(cudaStreamDestroy(nonBlockingStream1));
	CUDA_CALL(cudaStreamDestroy(nonBlockingStream2));
	CUDA_CALL(cudaEventDestroy(blockingEvent1));
	CUDA_CALL(cudaEventDestroy(blockingEvent2));

	for (size_t i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
	{
		CUDA_CALL(cudaStreamDestroy(neighborStreamVec[i]));
		CUDA_CALL(cudaEventDestroy(neighborEventVec[i]));
	}
}

void Simulator::run()
{
	auto getVolumeOfBubbles = [this]() -> double {
		KernelSize kernelSize(128, numBubbles);

		KERNEL_LAUNCH(calculateVolumes, kernelSize, 0, 0,
					  adp.r, adp.dummy1, numBubbles);

		return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, numBubbles);
	};

	std::cout << "======\nSetup\n======" << std::endl;
	{
		setupSimulation();
		saveSnapshotToFile();

		std::cout << "Letting bubbles settle after they've been created and before scaling or stabilization." << std::endl;
		stabilize();
		saveSnapshotToFile();

		const double phiTarget = properties.getPhiTarget();
		double bubbleVolume = getVolumeOfBubbles();
		double phi = bubbleVolume / properties.getSimulationBoxVolume();

		std::cout << "Volume ratios: current: " << phi
				  << ", target: " << phiTarget
				  << std::endl;

		std::cout << "Scaling the simulation box." << std::endl;
		transformPositions(true);
		const dvec relativeSize = properties.getBoxRelativeDimensions();
#if (NUM_DIM == 3)
		const double t = std::cbrt(getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z));
#else
		const double t = std::sqrt(getVolumeOfBubbles() / (phiTarget * relativeSize.x * relativeSize.y));
#endif
		properties.setTfr(dvec(t, t, t) * relativeSize);
		transformPositions(false);

		phi = bubbleVolume / properties.getSimulationBoxVolume();

		std::cout << "Volume ratios: current: " << phi
				  << ", target: " << phiTarget
				  << std::endl;

		saveSnapshotToFile();
	}

	std::cout << "=============\nStabilization\n=============" << std::endl;
	{
		int numSteps = 0;
		const int failsafe = 500;

		std::cout << "#steps\tdE/t\te1\te2" << std::endl;
		while (true)
		{
			double time = stabilize();
			double deltaEnergy = std::abs(energy2 - energy1) / time;
			deltaEnergy *= 0.5 * properties.getSigmaZero();

			if (deltaEnergy < properties.getMaxDeltaEnergy())
			{
				std::cout << "Final delta energy " << deltaEnergy
						  << " after " << (numSteps + 1) * properties.getNumStepsToRelax()
						  << " steps."
						  << " Energy before: " << energy1
						  << ", energy after: " << energy2
						  << ", time: " << time * timeScalingFactor
						  << std::endl;
				break;
			}
			else if (numSteps > failsafe)
			{
				std::cout << "Over " << failsafe * properties.getNumStepsToRelax()
						  << " steps taken and required delta energy not reached."
						  << " Check parameters."
						  << std::endl;
				break;
			}
			else
			{
				std::cout
					<< (numSteps + 1) * properties.getNumStepsToRelax() << "\t"
					<< deltaEnergy << "\t"
					<< energy1 << "\t"
					<< energy2
					<< std::endl;
			}

			++numSteps;
		}

		saveSnapshotToFile();
	}

	std::cout << "==========\nSimulation\n==========" << std::endl;
	std::stringstream dataStream;
	{
		// Set starting positions and reset wrapMultipliers to 0
		const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
		CUDA_CALL(cudaMemcpy(adp.x0, adp.x, numBytesToCopy, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemset(wrapMultipliers.get(), 0, wrapMultipliers.getSizeInBytes()));

		simulationTime = 0;
		int timesPrinted = 1;
		uint32_t numSteps = 0;
		const dvec interval = properties.getTfr() - properties.getLbb();

		KernelSize kernelSize(128, numBubbles);

		// Calculate the energy at simulation start
		KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
					  0.0, numBubbles, adp.dummy4);

		KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0,
					  numBubbles,
					  pairs.getRowPtr(0),
					  pairs.getRowPtr(1),
					  adp.r,
					  adp.dummy4,
					  interval.x, PBC_X == 1, adp.x,
					  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
					  ,
					  interval.z, PBC_Z == 1, adp.z
#endif
		);

		energy1 = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, numBubbles);

		// Start the simulation proper
		std::cout << "T\tR\t#b\tdE\t#steps" << std::endl;
		while (integrate())
		{
#if (USE_PROFILING == 1)
			if (numSteps == 2000)
				CUDA_PROFILER_START();
			else if (numSteps == 2050)
			{
				CUDA_PROFILER_STOP();
				break;
			}
#endif
			// The if clause contains many slow operations, but it's only done
			// very few times relative to the entire run time, so it should not
			// have a huge cost. Roughly 6e4-1e5 integration steps are taken for each time step
			// and the if clause is executed once per time step.
			const double scaledTime = simulationTime * timeScalingFactor;
			if ((int)scaledTime >= timesPrinted)
			{
				// Calculate total energy
				KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
							  0.0, numBubbles, adp.dummy4);

				KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0,
							  numBubbles,
							  pairs.getRowPtr(0),
							  pairs.getRowPtr(1),
							  adp.r,
							  adp.dummy4,
							  interval.x, PBC_X == 1, adp.x,
							  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
							  ,
							  interval.z, PBC_Z == 1, adp.z
#endif
				);

				energy2 = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, numBubbles);
				const double dE = (energy2 - energy1) / energy2;

				// Add values to data stream
				double relativeRadius = getAverageProperty(adp.r) / properties.getAvgRad();
				dataStream << scaledTime
						   << " " << relativeRadius
						   << " " << maxBubbleRadius / properties.getAvgRad()
						   << " " << numBubbles
						   << " " << getAverageProperty(adp.s)
						   << " " << getAverageProperty(adp.d)
						   << " " << dE
						   << "\n";

				// Print some values
				std::cout << scaledTime << "\t"
						  << relativeRadius << "\t"
						  << numBubbles << "\t"
						  << dE << "\t"
						  << numSteps
						  << std::endl;

				// Only write snapshots when t* is a power of 2.
				if ((timesPrinted & (timesPrinted - 1)) == 0)
					saveSnapshotToFile();

				++timesPrinted;
				numSteps = 0;
				energy1 = energy2;
			}

			++numSteps;
		}
	}

	std::ofstream file(properties.getDataFilename());
	file << dataStream.str() << std::endl;
}

void Simulator::setupSimulation()
{
	generateBubbles();

	const int numBubblesAboveMinRad = cubWrapper->reduce<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), numBubbles);
	if (numBubblesAboveMinRad < numBubbles)
		deleteSmallBubbles(numBubblesAboveMinRad);

	updateCellsAndNeighbors();

	// Calculate some initial values which are needed
	// for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
	const dvec tfr = properties.getTfr();
	const dvec lbb = properties.getLbb();
	const dvec interval = tfr - lbb;
	double timeStep = properties.getTimeStep();

	KernelSize kernelSize(128, numBubbles);

	KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
				  0.0, numBubbles,
				  adp.dxdtO,
				  adp.dydtO,
				  adp.dzdtO,
				  adp.drdtO,
				  adp.d,
				  adp.s);

	std::cout << "Calculating some initial values as a part of setup." << std::endl;

	KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0,
				  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r,
				  interval.x, lbb.x, PBC_X == 1, adp.x, adp.dxdtO,
				  interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO
#if (NUM_DIM == 3)
				  ,
				  interval.z, lbb.z, PBC_Z == 1, adp.z, adp.dzdtO
#endif
	);

	KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0,
				  numBubbles, timeStep,
				  adp.x, adp.dxdtO,
				  adp.y, adp.dydtO
#if (NUM_DIM == 3)
				  ,
				  adp.z, adp.dzdtO
#endif
	);

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
	KERNEL_LAUNCH(boundaryWrapKernel, kernelSize, 0, 0,
				  numBubbles
#if (PBC_X == 1)
				  ,
				  adp.x, lbb.x, tfr.x, wrapMultipliers.getRowPtr(3)
#endif
#if (PBC_Y == 1)
										   ,
				  adp.y, lbb.y, tfr.y, wrapMultipliers.getRowPtr(4)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
										   ,
				  adp.z, lbb.z, tfr.z, wrapMultipliers.getRowPtr(5)
#endif
	);
#endif

	KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
				  0.0, numBubbles,
				  adp.dxdtO, adp.dydtO, adp.dzdtO, adp.drdtO);

	KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0,
				  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.r,
				  interval.x, lbb.x, PBC_X == 1, adp.x, adp.dxdtO,
				  interval.y, lbb.y, PBC_Y == 1, adp.y, adp.dydtO
#if (NUM_DIM == 3)
				  ,
				  interval.z, lbb.z, PBC_Z == 1, adp.z, adp.dzdtO
#endif
	);
}

double Simulator::stabilize()
{
	// This function integrates only the positions of the bubbles.
	// Gas exchange is not used. This is used for equilibrating the foam.
	double elapsedTime = 0.0;

	KernelSize kernelSize(128, numBubbles);

	const dvec tfr = properties.getTfr();
	const dvec lbb = properties.getLbb();
	const dvec interval = tfr - lbb;

	double timeStep = properties.getTimeStep();
	double error = 100000;

	// This is a relatively heavy weight operation, but this function is not called too many times
	// so the cost should be negligible.
	cudaEvent_t energyEvent;
	cudaStream_t energyStream;
	CUDA_ASSERT(cudaStreamCreateWithFlags(&energyStream, cudaStreamNonBlocking));
	CUDA_ASSERT(cudaEventCreateWithFlags(&energyEvent, cudaEventBlockingSync));

	// Energy before stabilization
	{
		KERNEL_LAUNCH(resetKernel, kernelSize, 0, energyStream,
					  0.0, numBubbles, adp.dummy4);

		KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, energyStream,
					  numBubbles,
					  pairs.getRowPtr(0),
					  pairs.getRowPtr(1),
					  adp.r,
					  adp.dummy4,
					  interval.x, PBC_X == 1, adp.x,
					  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
					  ,
					  interval.z, PBC_Z == 1, adp.z
#endif
		);

		cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, dtfapr, numBubbles, energyStream);
		CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&pinnedDouble.get()[1]), static_cast<void *>(dtfapr), sizeof(double), cudaMemcpyDeviceToHost, energyStream));
		CUDA_CALL(cudaEventRecord(energyEvent, energyStream));
	}

	for (int i = 0; i < properties.getNumStepsToRelax(); ++i)
	{
		do
		{
			// Reset values to zero
			{
				KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
							  0.0, numBubbles,
							  adp.dxdtP,
							  adp.dydtP,
							  adp.dzdtP,
							  adp.error,
							  adp.dummy1,
							  adp.dummy2);
			}

			// Predict
			{
				KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0,
							  numBubbles, timeStep,
							  adp.xP, adp.x, adp.dxdt, adp.dxdtO,
							  adp.yP, adp.y, adp.dydt, adp.dydtO
#if (NUM_DIM == 3)
							  ,
							  adp.zP, adp.z, adp.dzdt, adp.dzdtO
#endif
				);

				CUDA_CALL(cudaEventRecord(blockingEvent2, 0));
			}

			// Velocity
			{
				KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0,
							  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP,
							  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP,
							  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#if (NUM_DIM == 3)
							  ,
							  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
				);

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
				KERNEL_LAUNCH(velocityWallKernel, pairKernelSize, 0, 0,
							  numBubbles, properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP
#if (PBC_X == 0)
							  ,
							  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP
#endif
#if (PBC_Y == 0)
							  ,
							  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#endif
#if (NUM_DIM == 3 && PBC_Z == 0)
							  ,
							  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
				);
#endif
			}

			// Correction
			{
				KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0,
							  numBubbles, timeStep, adp.error,
							  adp.xP, adp.x, adp.dxdt, adp.dxdtP,
							  adp.yP, adp.y, adp.dydt, adp.dydtP
#if (NUM_DIM == 3)
							  ,
							  adp.zP, adp.z, adp.dzdt, adp.dzdtP
#endif
				);

				CUDA_CALL(cudaEventRecord(blockingEvent2, 0));
				CUDA_CALL(cudaStreamWaitEvent(nonBlockingStream2, blockingEvent2, 0));
			}

			// Error
			{
				cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, adp.error, dtfa, numBubbles, nonBlockingStream2);
				CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), static_cast<void *>(dtfa), sizeof(double), cudaMemcpyDeviceToHost, nonBlockingStream2));
				CUDA_CALL(cudaEventRecord(blockingEvent2, nonBlockingStream2));
				CUDA_CALL(cudaEventSynchronize(blockingEvent2));

				error = pinnedDouble.get()[0];

				if (error < properties.getErrorTolerance() && timeStep < 0.1)
					timeStep *= 1.9;
				else if (error > properties.getErrorTolerance())
					timeStep *= 0.5;
			}

		} while (error > properties.getErrorTolerance());

		// Boundary wrap
		{
#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
			KERNEL_LAUNCH(boundaryWrapKernel, kernelSize, 0, 0,
						  numBubbles
#if (PBC_X == 1)
						  ,
						  adp.xP, lbb.x, tfr.x, wrapMultipliers.getRowPtr(0)
#endif
#if (PBC_Y == 1)
													,
						  adp.yP, lbb.y, tfr.y, wrapMultipliers.getRowPtr(1)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
													,
						  adp.zP, lbb.z, tfr.z, wrapMultipliers.getRowPtr(2)
#endif
			);
#endif
		}

		// Update the current values with the calculated predictions
		{
			CUDA_CALL(cudaStreamWaitEvent(0, energyEvent, 0));

			const size_t numBytesToCopy = 3 * sizeof(double) * dataStride;
			CUDA_CALL(cudaMemcpyAsync(adp.dxdtO, adp.dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
			CUDA_CALL(cudaMemcpyAsync(adp.x, adp.xP, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
			CUDA_CALL(cudaMemcpyAsync(adp.dxdt, adp.dxdtP, numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
		}

		properties.setTimeStep(timeStep);
		elapsedTime += timeStep;

		CUDA_CALL(cudaEventRecord(blockingEvent1, 0));
		CUDA_CALL(cudaStreamWaitEvent(0, blockingEvent1, 0));

		if (i % 50 == 0)
			updateCellsAndNeighbors();
	}

	// Energy after stabilization
	{
		CUDA_CALL(cudaEventSynchronize(energyEvent));
		energy1 = pinnedDouble.get()[1];

		KERNEL_LAUNCH(resetKernel, kernelSize, 0, energyStream,
					  0.0, numBubbles, adp.dummy4);

		CUDA_CALL(cudaStreamWaitEvent(energyStream, blockingEvent1, 0));
		KERNEL_LAUNCH(potentialEnergyKernel, pairKernelSize, 0, 0,
					  numBubbles,
					  pairs.getRowPtr(0),
					  pairs.getRowPtr(1),
					  adp.r,
					  adp.dummy4,
					  interval.x, PBC_X == 1, adp.x,
					  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
					  ,
					  interval.z, PBC_Z == 1, adp.z
#endif
		);

		cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy4, dta, numBubbles, energyStream);
		CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&pinnedDouble.get()[2]), static_cast<void *>(dta), sizeof(double), cudaMemcpyDeviceToHost, energyStream));
		CUDA_CALL(cudaEventRecord(energyEvent, energyStream));

		CUDA_CALL(cudaEventSynchronize(energyEvent));
		energy2 = pinnedDouble.get()[2];
	}

	CUDA_CALL(cudaStreamDestroy(energyStream));
	CUDA_CALL(cudaEventDestroy(energyEvent));

	return elapsedTime;
}

bool Simulator::integrate()
{
	KernelSize kernelSize(128, numBubbles);

	const dvec tfr = properties.getTfr();
	const dvec lbb = properties.getLbb();
	const dvec interval = tfr - lbb;

	double timeStep = properties.getTimeStep();
	double error = 100000;
	uint32_t numLoopsDone = 0;

	do
	{
		NVTX_RANGE_PUSH_A("Integration step");

		// Reset
		{
			KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0,
						  0.0, numBubbles,
						  adp.dxdtP,
						  adp.dydtP,
						  adp.dzdtP,
						  adp.drdtP,
						  adp.error,
						  adp.dummy1,
						  adp.dummy2);
		}

		// Predict
		{
			KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0,
						  numBubbles, timeStep,
						  adp.xP, adp.x, adp.dxdt, adp.dxdtO,
						  adp.yP, adp.y, adp.dydt, adp.dydtO,
#if (NUM_DIM == 3)
						  adp.zP, adp.z, adp.dzdt, adp.dzdtO,
#endif
						  adp.rP, adp.r, adp.drdt, adp.drdtO);

			CUDA_CALL(cudaEventRecord(blockingEvent2, 0));
		}

		// Velocity
		{
			KERNEL_LAUNCH(velocityPairKernel, pairKernelSize, 0, 0,
						  properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP,
						  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP,
						  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#if (NUM_DIM == 3)
						  ,
						  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
			);

#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
			KERNEL_LAUNCH(velocityWallKernel, pairKernelSize, 0, 0,
						  numBubbles, properties.getFZeroPerMuZero(), pairs.getRowPtr(0), pairs.getRowPtr(1), adp.rP
#if (PBC_X == 0)
						  ,
						  interval.x, lbb.x, PBC_X == 1, adp.xP, adp.dxdtP
#endif
#if (PBC_Y == 0)
						  ,
						  interval.y, lbb.y, PBC_Y == 1, adp.yP, adp.dydtP
#endif
#if (NUM_DIM == 3 && PBC_Z == 0)
						  ,
						  interval.z, lbb.z, PBC_Z == 1, adp.zP, adp.dzdtP
#endif
			);
#endif

#if USE_FLOW
			int *numNeighbors = bubbleCellIndices.getRowPtr(0);

			KERNEL_LAUNCH(neighborVelocityKernel, pairKernelSize, 0, 0,
						  pairs.getRowPtr(0), pairs.getRowPtr(1), numNeighbors,
						  adp.dummy1, adp.dxdtO,
						  adp.dummy2, adp.dydtO
#if (NUM_DIM == 3)
						  ,
						  adp.dummy3, adp.dzdtO
#endif
			);

			KERNEL_LAUNCH(flowVelocityKernel, pairKernelSize, 0, 0,
						  numBubbles, numNeighbors,
						  adp.dxdtP, adp.dydtP, adp.dzdtP,
						  adp.dummy1, adp.dummy2, adp.dummy3,
						  adp.xP, adp.yP, adp.zP,
						  properties.getFlowVel(),
						  properties.getFlowTfr(),
						  properties.getFlowLbb());
#endif

			KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0,
						  numBubbles, timeStep, adp.error,
						  adp.xP, adp.x, adp.dxdt, adp.dxdtP,
						  adp.yP, adp.y, adp.dydt, adp.dydtP
#if (NUM_DIM == 3)
						  ,
						  adp.zP, adp.z, adp.dzdt, adp.dzdtP,
#endif
			);
		}

		// Gas exchange
		{
			KernelSize kernelSize(128, numBubbles);

			CUDA_CALL(cudaStreamWaitEvent(nonBlockingStream2, blockingEvent2, 0));

			KERNEL_LAUNCH(gasExchangeKernel, pairKernelSize, 0, nonBlockingStream2,
						  numBubbles,
						  pairs.getRowPtr(0),
						  pairs.getRowPtr(1),
						  adp.rP,
						  adp.drdtP,
						  adp.dummy1,
						  interval.x, PBC_X == 1, adp.xP,
						  interval.y, PBC_Y == 1, adp.yP
#if (NUM_DIM == 3)
						  ,
						  interval.z, PBC_Z == 1, adp.zP
#endif
			);

			KERNEL_LAUNCH(freeAreaKernel, kernelSize, 0, nonBlockingStream2,
						  numBubbles, adp.rP, adp.dummy1, adp.dummy2, adp.dummy3);

			cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtfa, numBubbles, nonBlockingStream2);
			cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy2, dtfapr, numBubbles, nonBlockingStream2);
			cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy3, dta, numBubbles, nonBlockingStream2);

			KERNEL_LAUNCH(finalRadiusChangeRateKernel, kernelSize, 0, nonBlockingStream2,
						  adp.drdtP, adp.rP, adp.dummy1, numBubbles, properties.getKappa(), properties.getKParameter());

			KERNEL_LAUNCH(correctKernel, kernelSize, 0, nonBlockingStream2,
						  numBubbles, timeStep, adp.error,
						  adp.rP, adp.r, adp.drdt, adp.drdtP);

			CUDA_CALL(cudaEventRecord(blockingEvent2, nonBlockingStream2));
			CUDA_CALL(cudaStreamWaitEvent(0, blockingEvent2, 0));
		}

		// Error
		{
			error = cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Max, adp.error, numBubbles);

			if (error < properties.getErrorTolerance() && timeStep < 0.1)
				timeStep *= 1.9;
			else if (error > properties.getErrorTolerance())
				timeStep *= 0.5;
		}

		++numLoopsDone;

		NVTX_RANGE_POP();
	} while (error > properties.getErrorTolerance());

	// Path lengths & distances
	{
		KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, 0,
					  numBubbles,
					  adp.s,
					  adp.d,
					  adp.xP, adp.x, adp.x0, wrapMultipliers.getRowPtr(0), interval.x,
					  adp.yP, adp.y, adp.y0, wrapMultipliers.getRowPtr(1), interval.y
#if (NUM_DIM == 3)
					  ,
					  adp.zP, adp.z, adp.z0, wrapMultipliers.getRowPtr(2), interval.z
#endif
		);
	}

	// Boundary wrap
	{
#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
		KERNEL_LAUNCH(boundaryWrapKernel, kernelSize, 0, 0,
					  numBubbles
#if (PBC_X == 1)
					  ,
					  adp.xP, lbb.x, tfr.x, wrapMultipliers.getRowPtr(0)
#endif
#if (PBC_Y == 1)
												,
					  adp.yP, lbb.y, tfr.y, wrapMultipliers.getRowPtr(1)
#endif
#if (PBC_Z == 1 && NUM_DIM == 3)
												,
					  adp.zP, lbb.z, tfr.z, wrapMultipliers.getRowPtr(2)
#endif
		);
#endif
	}

	// Calculate how many bubbles are below the minimum size
	{
		const cudaStream_t originalStream = 0;

		KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, kernelSize, 0, nonBlockingStream1,
					  numBubbles,
					  aboveMinRadFlags.getRowPtr(0),
					  adp.rP,
					  properties.getMinRad());

		cubWrapper->reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, aboveMinRadFlags.getRowPtr(0), static_cast<int *>(mbpc), numBubbles, nonBlockingStream1);
		CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedInt.get()), mbpc, sizeof(int), cudaMemcpyDeviceToHost, nonBlockingStream1));

		cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, adp.rP, static_cast<double *>(dtfa), numBubbles, nonBlockingStream1);
		CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(pinnedDouble.get()), dtfa, sizeof(double), cudaMemcpyDeviceToHost, nonBlockingStream1));

		CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream1));
	}

	// Update values
	{
		CUDA_CALL(cudaStreamWaitEvent(0, blockingEvent1, 0));
		const size_t numBytesToCopy = 4 * sizeof(double) * dataStride;

		CUDA_CALL(cudaMemcpyAsync(adp.dxdtO, adp.dxdt, numBytesToCopy, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpyAsync(adp.x, adp.xP, 2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
	}

	++integrationStep;
	properties.setTimeStep(timeStep);
	simulationTime += timeStep;

	CUDA_CALL(cudaEventSynchronize(blockingEvent1));
	maxBubbleRadius = pinnedDouble.get()[0];

	// Delete & reorder
	{
		const int numBubblesAboveMinRad = pinnedInt.get()[0];
		const bool shouldDeleteBubbles = numBubblesAboveMinRad < numBubbles;

		if (shouldDeleteBubbles)
			deleteSmallBubbles(numBubblesAboveMinRad);

		if (shouldDeleteBubbles || integrationStep % 50 == 0)
			updateCellsAndNeighbors();
	}

	bool continueSimulation = numBubbles > properties.getMinNumBubbles();
#if (NUM_DIM == 3)
	continueSimulation &= maxBubbleRadius < 0.5 * (tfr - lbb).getMinComponent();
#endif

	return continueSimulation;
}

void Simulator::generateBubbles()
{
	std::cout << "Starting to generate data for bubbles." << std::endl;

	const int rngSeed = properties.getRngSeed();
	const double avgRad = properties.getAvgRad();
	const double stdDevRad = properties.getStdDevRad();
	const dvec tfr = properties.getTfr();
	const dvec lbb = properties.getLbb();

	curandGenerator_t generator;
	CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));

	CURAND_CALL(curandGenerateUniformDouble(generator, adp.x, numBubbles));
	CURAND_CALL(curandGenerateUniformDouble(generator, adp.y, numBubbles));
#if (NUM_DIM == 3)
	CURAND_CALL(curandGenerateUniformDouble(generator, adp.z, numBubbles));
#endif
	CURAND_CALL(curandGenerateUniformDouble(generator, adp.rP, numBubbles));
	CURAND_CALL(curandGenerateNormalDouble(generator, adp.r, numBubbles, avgRad, stdDevRad));

	CURAND_CALL(curandDestroyGenerator(generator));

	KernelSize kernelSize(128, numBubbles);

	assert(bubblesPerDimAtStart.x > 0);
	assert(bubblesPerDimAtStart.y > 0);
#if (NUM_DIM == 3)
	assert(bubblesPerDimAtStart.z > 0);
#endif
	KERNEL_LAUNCH(assignDataToBubbles, kernelSize, 0, 0,
				  adp.x, adp.y, adp.z, adp.xP, adp.yP, adp.zP, adp.r, adp.rP,
				  aboveMinRadFlags.getRowPtr(0), bubblesPerDimAtStart,
				  tfr, lbb, avgRad, properties.getMinRad(), numBubbles);

	cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.rP, dasai, numBubbles, 0);
	CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.rP), static_cast<void *>(adp.r),
							  sizeof(double) * dataStride,
							  cudaMemcpyDeviceToDevice,
							  0));
}

void Simulator::updateCellsAndNeighbors()
{
	dim3 gridSize = getGridSize();
	const int numCells = gridSize.x * gridSize.y * gridSize.z;
	const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

	int *offsets = cellData.getRowPtr((size_t)CellProperty::OFFSET);
	int *sizes = cellData.getRowPtr((size_t)CellProperty::SIZE);

	cellData.setBytesToZero();
	bubbleCellIndices.setBytesToZero();

	KernelSize kernelSize(128, numBubbles);

	KERNEL_LAUNCH(assignBubblesToCells, pairKernelSize, 0, 0,
				  adp.x, adp.y, adp.z,
				  bubbleCellIndices.getRowPtr(2), bubbleCellIndices.getRowPtr(3),
				  properties.getLbb(), properties.getTfr(), cellDim, numBubbles);

	int *cellIndices = bubbleCellIndices.getRowPtr(0);
	int *bubbleIndices = bubbleCellIndices.getRowPtr(1);

	cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
									const_cast<const int *>(bubbleCellIndices.getRowPtr(2)),
									cellIndices,
									const_cast<const int *>(bubbleCellIndices.getRowPtr(3)),
									bubbleIndices,
									numBubbles);

	CUDA_CALL(cudaEventRecord(blockingEvent1));
	CUDA_CALL(cudaStreamWaitEvent(nonBlockingStream1, blockingEvent1, 0));

	cubWrapper->histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven,
												bubbleCellIndices.getRowPtr(2),
												sizes,
												numCells + 1,
												0,
												numCells,
												numBubbles);

	cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, numCells);
	CUDA_CALL(cudaEventRecord(blockingEvent2));

	KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, nonBlockingStream1,
				  numBubbles, ReorganizeType::COPY_FROM_INDEX, bubbleIndices, bubbleIndices,
				  adp.x, adp.xP,
				  adp.y, adp.yP,
				  adp.z, adp.zP,
				  adp.r, adp.rP,
				  adp.dxdt, adp.dxdtP,
				  adp.dydt, adp.dydtP,
				  adp.dzdt, adp.dzdtP,
				  adp.drdt, adp.drdtP,
				  adp.dxdtO, adp.error,
				  adp.dydtO, adp.dummy1,
				  adp.dzdtO, adp.dummy2,
				  adp.drdtO, adp.dummy3,
				  adp.x0, adp.dummy4,
				  adp.y0, adp.dummy5,
				  adp.z0, adp.dummy6,
				  adp.s, adp.dummy7,
				  adp.d, adp.dummy8,
				  wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(3),
				  wrapMultipliers.getRowPtr(1), wrapMultipliers.getRowPtr(4),
				  wrapMultipliers.getRowPtr(2), wrapMultipliers.getRowPtr(5));

	CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x),
							  static_cast<void *>(adp.xP),
							  sizeof(double) * numAliases / 2 * dataStride,
							  cudaMemcpyDeviceToDevice,
							  nonBlockingStream1));

	CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrapMultipliers.getRowPtr(0)),
							  static_cast<void *>(wrapMultipliers.getRowPtr(3)),
							  wrapMultipliers.getSizeInBytes() / 2,
							  cudaMemcpyDeviceToDevice,
							  nonBlockingStream1));

	CUDA_CALL(cudaEventRecord(blockingEvent1, nonBlockingStream1));

	dvec interval = properties.getTfr() - properties.getLbb();

	kernelSize.block = dim3(128, 1, 1);
	kernelSize.grid = gridSize;

	CUDA_CALL(cudaMemset(np, 0, sizeof(int)));

	for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
	{
		CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], blockingEvent1, 0));
		CUDA_CALL(cudaStreamWaitEvent(neighborStreamVec[i], blockingEvent2, 0));

		KERNEL_LAUNCH(neighborSearch, kernelSize, 0, neighborStreamVec[i],
					  i, numBubbles, numCells, static_cast<int>(pairs.getWidth()),
					  offsets, sizes, pairs.getRowPtr(2), pairs.getRowPtr(3), adp.r,
					  interval.x, PBC_X == 1, adp.x,
					  interval.y, PBC_Y == 1, adp.y
#if (NUM_DIM == 3)
					  ,
					  interval.z, PBC_Z == 1, adp.z
#endif
		);

		CUDA_CALL(cudaEventRecord(neighborEventVec[i], neighborStreamVec[i]));
		CUDA_CALL(cudaStreamWaitEvent(0, neighborEventVec[i], 0));
	}

	CUDA_CALL(cudaMemcpy(static_cast<void *>(pinnedInt.get()), np, sizeof(int), cudaMemcpyDeviceToHost));
	int numPairs = pinnedInt.get()[0];
	cubWrapper->sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs,
									const_cast<const int *>(pairs.getRowPtr(2)),
									pairs.getRowPtr(0),
									const_cast<const int *>(pairs.getRowPtr(3)),
									pairs.getRowPtr(1),
									numPairs);
}

void Simulator::deleteSmallBubbles(int numBubblesAboveMinRad)
{
	NVTX_RANGE_PUSH_A("BubbleRemoval");
	KernelSize kernelSize(128, numBubbles);

	int *flag = aboveMinRadFlags.getRowPtr(0);

	CUDA_CALL(cudaMemset(static_cast<void *>(dvm), 0, sizeof(double)));
	KERNEL_LAUNCH(calculateRedistributedGasVolume, kernelSize, 0, 0,
				  adp.dummy1, adp.r, flag, numBubbles);

	cubWrapper->reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, adp.dummy1, dtv, numBubbles);

	int *newIdx = aboveMinRadFlags.getRowPtr(1);
	cubWrapper->scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, flag, newIdx, numBubbles);

	KERNEL_LAUNCH(reorganizeKernel, kernelSize, 0, 0,
				  numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx, flag,
				  adp.x, adp.xP,
				  adp.y, adp.yP,
				  adp.z, adp.zP,
				  adp.r, adp.rP,
				  adp.dxdt, adp.dxdtP,
				  adp.dydt, adp.dydtP,
				  adp.dzdt, adp.dzdtP,
				  adp.drdt, adp.drdtP,
				  adp.dxdtO, adp.error,
				  adp.dydtO, adp.dummy1,
				  adp.dzdtO, adp.dummy2,
				  adp.drdtO, adp.dummy3,
				  adp.x0, adp.dummy4,
				  adp.y0, adp.dummy5,
				  adp.z0, adp.dummy6,
				  adp.s, adp.dummy7,
				  adp.d, adp.dummy8,
				  wrapMultipliers.getRowPtr(0), wrapMultipliers.getRowPtr(3),
				  wrapMultipliers.getRowPtr(1), wrapMultipliers.getRowPtr(4),
				  wrapMultipliers.getRowPtr(2), wrapMultipliers.getRowPtr(5));

	CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(adp.x),
							  static_cast<void *>(adp.xP),
							  sizeof(double) * numAliases / 2 * dataStride,
							  cudaMemcpyDeviceToDevice));

	CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(wrapMultipliers.getRowPtr(0)),
							  static_cast<void *>(wrapMultipliers.getRowPtr(3)),
							  wrapMultipliers.getSizeInBytes() / 2,
							  cudaMemcpyDeviceToDevice));

	numBubbles = numBubblesAboveMinRad;
	KERNEL_LAUNCH(addVolume, kernelSize, 0, 0,
				  adp.r, numBubbles);

	NVTX_RANGE_POP();
}

dim3 Simulator::getGridSize()
{
	const int totalNumCells = std::ceil((float)numBubbles / properties.getNumBubblesPerCell());
	dvec interval = properties.getTfr() - properties.getLbb();
	interval /= interval.x;
#if (NUM_DIM == 3)
	float nx = std::cbrt((float)totalNumCells / (interval.y * interval.z));
#else
	float nx = std::sqrt((float)totalNumCells / interval.y);
	interval.z = 0;
#endif
	ivec grid = (nx * interval).floor() + 1;
	assert(grid.x > 0);
	assert(grid.y > 0);
	assert(grid.z > 0);

	return dim3(grid.x, grid.y, grid.z);
}

void Simulator::transformPositions(bool normalize)
{
	KERNEL_LAUNCH(transformPositionsKernel, pairKernelSize, 0, 0,
				  normalize, numBubbles, properties.getLbb(), properties.getTfr(),
				  adp.x,
				  adp.y,
				  adp.z);
}

double Simulator::getAverageProperty(double *p)
{
	return cubWrapper->reduce<double, double *, double *>(&cub::DeviceReduce::Sum, p, numBubbles) / numBubbles;
}

void Simulator::saveSnapshotToFile()
{
	std::stringstream ss;
	ss << properties.getSnapshotFilename() << ".csv." << numSnapshots;
	std::ofstream file(ss.str().c_str(), std::ios::out);
	if (file.is_open())
	{
		const size_t numComp = 17;
		hostData.clear();
		hostData.resize(dataStride * numComp);
		CUDA_CALL(cudaMemcpy(hostData.data(), deviceData, sizeof(double) * numComp * dataStride, cudaMemcpyDeviceToHost));

		file << "x,y,z,r,vx,vy,vz,path,dist\n";
		for (size_t i = 0; i < (size_t)numBubbles; ++i)
		{
			file << hostData[i + 0 * dataStride];
			file << ",";
			file << hostData[i + 1 * dataStride];
			file << ",";
			file << hostData[i + 2 * dataStride];
			file << ",";
			file << hostData[i + 3 * dataStride];
			file << ",";
			file << hostData[i + 4 * dataStride];
			file << ",";
			file << hostData[i + 5 * dataStride];
			file << ",";
			file << hostData[i + 6 * dataStride];
			file << ",";
			file << hostData[i + 15 * dataStride];
			file << ",";
			file << hostData[i + 16 * dataStride];
			file << "\n";
		}

		++numSnapshots;
	}
}

} // namespace cubble
