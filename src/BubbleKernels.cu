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

__device__ void addVelocity(int idx1, int idx2, double multiplier, double maxDistance, bool shouldWrap, double *x, double *v)
{
	v[idx1] += getWrappedDistance(idx1, idx2, maxDistance, shouldWrap, x) * multiplier;
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