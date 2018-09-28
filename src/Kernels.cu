// -*- C++ -*-

#include "Kernels.cuh"

namespace cubble
{
__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum)
{
	// Switch statements and ifs that diverge inside one warp/block are
	// detrimental for performance. However, this should never diverge,
	// as all the threads of one block should always be in the same cell
	// going for the same neighbor.
	ivec idxVec = cellIdx;
	switch (neighborNum)
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

__device__ int getGlobalTid()
{
	// Simple helper function for calculating a 1D coordinate
	// from 1, 2 or 3 dimensional coordinates.
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int blocksBefore = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadsBefore = blockDim.y * blockDim.x * threadIdx.z + blockDim.x * threadIdx.y;
	int tid = blocksBefore * threadsPerBlock + threadsBefore + threadIdx.x;

	return tid;
}

__device__ double getWrappedCoordinate(double val1, double val2, double multiplier)
{
	double difference = val1 - val2;
	val2 = difference < -0.5 * multiplier ? val2 - multiplier : (difference > 0.5 * multiplier ? val2 + multiplier : val2);
	val2 = val1 - val2;

	return val2 * multiplier;
}

__device__ dvec getWrappedPos(dvec pos, dvec tfr, dvec lbb)
{
	const dvec interval = tfr - lbb;
	pos.x = pos.x < lbb.x ? pos.x + interval.x : (pos.x > tfr.x ? pos.x - interval.x : pos.x);
	pos.y = pos.y < lbb.y ? pos.y + interval.y : (pos.y > tfr.y ? pos.y - interval.y : pos.y);
	pos.z = pos.z < lbb.z ? pos.z + interval.z : (pos.z > tfr.z ? pos.z - interval.z : pos.z);

	return pos;
}

__device__ int getCellIdxFromPos(double x, double y, double z, dvec interval, ivec cellDim)
{
	const int xid = floor(cellDim.x * x / interval.x);
	const int yid = floor(cellDim.y * y / interval.y);
	const int zid = floor(cellDim.z * z / interval.z);

	return zid * cellDim.x * cellDim.y + yid * cellDim.x + xid;
}

__device__ void resetDoubleArrayToValue(double value, int tid, double *array)
{
    array[tid] = value;
}

__global__ void calculateVolumes(double *r, double *volumes, int numBubbles, double pi)
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

__global__ void assignDataToBubbles(double *x, double *y, double *z,
									double *xPrd, double *yPrd, double *zPrd,
									double *r,
									double *w,
									int *aboveMinRadFlags,
									ivec bubblesPerDim,
									dvec tfr,
									dvec lbb,
									double avgRad,
									double minRad,
									int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		dvec pos(0, 0, 0);
		pos.x = (tid % bubblesPerDim.x) / (double)bubblesPerDim.x;
		pos.y = ((tid / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

		dvec randomOffset(x[tid], y[tid], 0);
#if (NUM_DIM == 3)
		randomOffset.z = z[tid];
		pos.z = (tid / (bubblesPerDim.x * bubblesPerDim.y))  / (double)bubblesPerDim.z;
#endif
		pos *= tfr - lbb;
		randomOffset = dvec::normalize(randomOffset) * avgRad * w[tid];
		pos = getWrappedPos(pos + randomOffset, tfr, lbb);

		x[tid] = pos.x;
		y[tid] = pos.y;
		z[tid] = pos.z;

		xPrd[tid] = pos.x;
		yPrd[tid] = pos.y;
		zPrd[tid] = pos.z;

		double radius = r[tid];
		r[tid] = radius > 0 ? radius : -radius;
		w[tid] = r[tid];
		aboveMinRadFlags[tid] = radius < minRad ? 0 : 1;
	}
}

__global__ void findOffsets(int *cellIndices, int *offsets, int numCells, int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		if (tid == 0)
			offsets[0] = 0;
		else
		{
			const int cellIdx = cellIndices[tid];
			if (cellIdx > cellIndices[tid - 1])
				offsets[cellIdx] = tid;
		}
	}
}

__global__ void findSizes(int *offsets, int *sizes, int numCells, int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numCells)
	{
		const int nextOffset = tid < numCells - 1 ? offsets[tid + 1] : numBubbles;
		sizes[tid] = nextOffset - offsets[tid];
	}
}

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec interval, ivec cellDim, int numBubbles)
{
	int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		const int cellIdx = getCellIdxFromPos(x[tid], y[tid], z[tid], interval, cellDim);
		cellIndices[tid] = cellIdx;
		bubbleIndices[tid] = tid;
	}
}

__global__ void findBubblePairs(double *x, double *y, double *z, double *r,
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

	const int selfCellIndex = blockIdx.z / numNeighborCells * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
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

			idx1 = selfOffset + idx1;
			idx2 = neighborOffset + idx2;

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

			if (magnitude < 1.1 * wrappedComponent)
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

__global__ void predict(double *x, double *y, double *z, double *r,
						double *xPrd, double *yPrd, double *zPrd, double *rPrd,
						double *dxdt, double *dydt, double *dzdt, double *drdt,
						double *dxdtOld, double *dydtOld, double *dzdtOld, double *drdtOld,
						dvec tfr,
						dvec lbb,
						double timeStep,
						int numBubbles,
						bool useGasExchange)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
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

		pos += 0.5 * timeStep * (3.0 * vel - velOld);
		pos = getWrappedPos(pos, tfr, lbb);

		xPrd[tid] = pos.x;
		yPrd[tid] = pos.y;
		zPrd[tid] = pos.z;

		if (useGasExchange)
			rPrd[tid] = r[tid] + 0.5 * timeStep * (3.0 * drdt[tid] - drdtOld[tid]);
	}
}

__global__ void calculateVelocityAndGasExchange(double *x, double *y, double *z, double *r,
												double *dxdt, double *dydt, double *dzdt, double *drdt,
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

		const double radii = r[idx1] + r[idx2];
		if (magnitude <= radii)
		{
			DEVICE_ASSERT(radii > 0);
			const double invRadii = 1.0 / radii;

			if (calculateEnergy)
			{
				double potentialEnergy = radii - magnitude;
				potentialEnergy *= potentialEnergy;
				atomicAdd(&energy[idx1], potentialEnergy);
				atomicAdd(&energy[idx2], potentialEnergy);
			}

			const double invMagnitude = 1.0 / magnitude;
			double generalVariable = fZeroPerMuZero * (radii - magnitude) * invRadii * invMagnitude;

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
				if (magnitude < velX || magnitude < velY)
				{
					velZ = velX < velY ? velX : velY;
					velZ *= velZ;
				}
				else
				{
					generalVariable = velY * velY;
					velZ = 0.5 * (generalVariable - velX * velX + magnitude * magnitude) * invMagnitude;
					velZ *= velZ;
					velZ = generalVariable - velZ;
					DEVICE_ASSERT(velZ > -0.0001);
					velZ = velZ < 0 ? -velZ : velZ;
					DEVICE_ASSERT(velZ >= 0);
				}

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
}

__global__ void calculateFreeAreaPerRadius(double *r, double *freeArea, double *output, double pi, int numBubbles)
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

__global__ void calculateFinalRadiusChangeRate(double *drdt, double *r, double *freeArea, int numBubbles, double invRho, double invPi, double kappa, double kParam)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		double invRadius = 1.0 / r[tid];
		double vr = kappa * freeArea[tid] * (invRho - invRadius);
		vr += drdt[tid];

		vr *= 0.5 * invPi * invRadius;
#if (NUM_DIM == 3)
		vr *= 0.5 * invRadius;
#endif

		drdt[tid] = kParam * vr;
	}
}

__global__ void correct(double *x, double *y, double *z, double *r,
						double *xPrd, double *yPrd, double *zPrd, double *rPrd,
						double *dxdt, double *dydt, double *dzdt, double *drdt,
						double *dxdtPrd, double *dydtPrd, double *dzdtPrd, double *drdtPrd,
						double *errors,
						int *aboveMinRadFlags,
						double minRad,
						dvec tfr,
						dvec lbb,
						double timeStep,
						int numBubbles,
						bool useGasExchange)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
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

		pos += 0.5 * timeStep * (vel + velPrd);
		pos = getWrappedPos(pos, tfr, lbb);

		double radError = 0;
		if (useGasExchange)
		{
			const double radius = r[tid] + 0.5 * timeStep * (drdt[tid] + drdtPrd[tid]);

			radError = radius - rPrd[tid];
			radError = radError < 0 ? -radError : radError;

			rPrd[tid] = radius;
			aboveMinRadFlags[tid] = radius < minRad ? 0 : 1;
		}
		else
			aboveMinRadFlags[tid] = 1;

		double error = (pos - posPrd).getAbsolute().getMaxComponent();
		error = error > radError ? error : radError;
		errors[tid] = error;

		xPrd[tid] = pos.x;
		yPrd[tid] = pos.y;
		zPrd[tid] = pos.z;
	}
}

__global__ void addVolume(double *r, double *volumeMultiplier, int numBubbles, double invTotalVolume)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		double multiplier = volumeMultiplier[0] * invTotalVolume;
		multiplier += 1.0;

#if (NUM_DIM == 3)
		multiplier = cbrt(multiplier);
#else
		multiplier = sqrt(multiplier);
#endif
		r[tid] = r[tid] * multiplier;
	}
}

__global__ void eulerIntegration(double *x, double *y, double *z, double *r,
								 double *dxdt, double *dydt, double *dzdt, double *drdt,
								 dvec tfr,
								 dvec lbb,
								 double timeStep,
								 int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		dvec pos(0, 0, 0);
		pos.x = x[tid];
		pos.y = y[tid];
		pos.z = z[tid];

		dvec vel(0, 0, 0);
		vel.x = dxdt[tid];
		vel.y = dydt[tid];
		vel.z = dzdt[tid];

		pos += timeStep * vel;
		pos = getWrappedPos(pos, tfr, lbb);

		x[tid] = pos.x;
		y[tid] = pos.y;
		z[tid] = pos.z;
		r[tid] = r[tid] + timeStep * drdt[tid];
	}
}

__global__ void calculateRedistributedGasVolume(double *volume, double *r, int *aboveMinRadFlags, double *volumeMultiplier, double pi, int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		const double radius = r[tid];
		double vol = pi * radius * radius;
#if (NUM_DIM == 3)
		vol *= 1.333333333333333333333333 * radius;
#endif

		if (aboveMinRadFlags[tid] == 0)
		{
			atomicAdd(volumeMultiplier, vol);
			volume[tid] = 0;
		}
		else
			volume[tid] = vol;
	}
}
} // namespace cubble