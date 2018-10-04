// -*- C++ -*-

#include "BubbleKernels.cuh"
#include "UtilityKernels.cuh"

namespace cubble
{
__device__ int dMaxBubblesPerCell;
__device__ double dTotalFreeArea;
__device__ double dTotalFreeAreaPerRadius;
__device__ double dVolumeMultiplier;
__device__ double dTotalVolume;

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum)
{
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

#if (PBC_X == 1)
	idxVec.x += dim.x;
	idxVec.x %= dim.x;
#else
	if (idxVec.x < 0)
		return -1;
#endif

#if (PBC_Y == 1)
	idxVec.y += dim.y;
	idxVec.y %= dim.y;
#else
	if (idxVec.y < 0)
		return -1;
#endif

#if (PBC_Z == 1)
	idxVec.z += dim.z;
	idxVec.z %= dim.z;
#else
	if (idxVec.z < 0)
		return -1;
#endif

	return get1DIdxFrom3DIdx(idxVec, dim);
}

__device__ double getWrappedCoordinate(double val1, double val2, double multiplier)
{
	double difference = val1 - val2;
	val2 = difference < -0.5 * multiplier ? val2 - multiplier : (difference > 0.5 * multiplier ? val2 + multiplier : val2);
	val2 = val1 - val2;

	return val2;
}

__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb, dvec tfr, ivec cellDim)
{
	const dvec interval = tfr - lbb;
	const int xid = floor(cellDim.x * (x - lbb.x) / interval.x);
	const int yid = floor(cellDim.y * (y - lbb.y) / interval.y);
	const int zid = floor(cellDim.z * (z - lbb.z) / interval.z);

	return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}
__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim)
{
	return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;
}

__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim)
{
	ivec idxVec(0, 0, 0);
	idxVec.x = idx % cellDim.x;
	idxVec.y = (idx / cellDim.x) % cellDim.y;
	idxVec.z = idx / (cellDim.x * cellDim.y);

	return idxVec;
}

__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue)
{
	const double interval = maxValue - minValue;
	double value = coordinate[idx];
	value = value < minValue ? value + interval : (value > maxValue ? value - interval : value);
	coordinate[idx] = value;
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
		pos.z = (tid / (bubblesPerDim.x * bubblesPerDim.y)) / (double)bubblesPerDim.z;
#endif
		pos *= tfr - lbb;
		randomOffset = dvec::normalize(randomOffset) * avgRad * w[tid];
		pos += randomOffset;

		x[tid] = pos.x;
		y[tid] = pos.y;
		z[tid] = pos.z;

		xPrd[tid] = pos.x;
		yPrd[tid] = pos.y;
		zPrd[tid] = pos.z;

		wrapAround(tid,
				   x, lbb.x, tfr.x,
				   y, lbb.y, tfr.y,
				   z, lbb.z, tfr.z,
				   xPrd, lbb.x, tfr.x,
				   yPrd, lbb.y, tfr.y,
				   zPrd, lbb.z, tfr.z);

		r[tid] = r[tid] > 0 ? r[tid] : -r[tid];
		w[tid] = r[tid];
		setFlagIfGreaterThanConstant(tid, aboveMinRadFlags, r, minRad);
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

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec lbb, dvec tfr, ivec cellDim, int numBubbles)
{
	int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		const int cellIdx = getCellIdxFromPos(x[tid], y[tid], z[tid], lbb, tfr, cellDim);
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

			idx1 = selfOffset + idx1;
			idx2 = neighborOffset + idx2;

			DEVICE_ASSERT(idx1 < numBubbles);
			DEVICE_ASSERT(idx2 < numBubbles);

			if (idx1 == idx2 || (selfComparison && idx2 < idx1))
				continue;

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

__global__ void calculateFinalRadiusChangeRate(double *drdt, double *r, double *freeArea, int numBubbles, double invPi, double kappa, double kParam)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		const double invRho = dTotalFreeAreaPerRadius / dTotalFreeArea;
		const double invRadius = 1.0 / r[tid];
		double vr = kappa * freeArea[tid] * (invRho - invRadius);
		vr += drdt[tid];

		vr *= 0.5 * invPi * invRadius;
#if (NUM_DIM == 3)
		vr *= 0.5 * invRadius;
#endif

		drdt[tid] = kParam * vr;
	}
}

__global__ void addVolume(double *r, int numBubbles)
{
	const int tid = getGlobalTid();
	if (tid < numBubbles)
	{
		double multiplier = dVolumeMultiplier / dTotalVolume;
		multiplier += 1.0;

#if (NUM_DIM == 3)
		multiplier = cbrt(multiplier);
#else
		multiplier = sqrt(multiplier);
#endif
		r[tid] = r[tid] * multiplier;
	}
}

__global__ void calculateRedistributedGasVolume(double *volume, double *r, int *aboveMinRadFlags, double pi, int numBubbles)
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
			atomicAdd(&dVolumeMultiplier, vol);
			volume[tid] = 0;
		}
		else
			volume[tid] = vol;
	}
}
} // namespace cubble