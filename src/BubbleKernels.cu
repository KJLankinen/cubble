// -*- C++ -*-

#include "BubbleKernels.cuh"
#include "UtilityKernels.cuh"

namespace cubble
{
__device__ int dMaxBubblesPerCell;
__device__ int dNumPairs;
__device__ double dTotalFreeArea;
__device__ double dTotalFreeAreaPerRadius;
__device__ double dVolumeMultiplier;
__device__ double dTotalVolume;

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum, bool wrapX, bool wrapY, bool wrapZ)
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

	if (wrapX)
	{
		idxVec.x += dim.x;
		idxVec.x %= dim.x;
	}
	else if (idxVec.x < 0 || idxVec.x >= dim.x)
		return -1;

	if (wrapY)
	{
		idxVec.y += dim.y;
		idxVec.y %= dim.y;
	}
	else if (idxVec.y < 0 || idxVec.y >= dim.y)
		return -1;

	if (wrapZ)
	{
		idxVec.z += dim.z;
		idxVec.z %= dim.z;
	}
	else if (idxVec.z < 0 || idxVec.z >= dim.z)
		return -1;

	return get1DIdxFrom3DIdx(idxVec, dim);
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

__device__ void addVelocity(int idx1, int idx2, double multiplier, double maxDistance, double minDistance, bool shouldWrap, double *x, double *v)
{
	const double velocity = getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap) * multiplier;
	atomicAdd(&v[idx1], velocity);
	atomicAdd(&v[idx2], -velocity);
}

__device__ void forceFromWalls(int idx, double fZeroPerMuZero, double *r,
							   double interval, double zeroPoint, bool shouldWrap, double *x, double *v)
{
	if (shouldWrap)
		return;

	const double radius = r[idx];
	const double distance1 = x[idx] - zeroPoint;
	const double distance2 = x[idx] - (interval + zeroPoint);
	double distance = distance1 * distance1 < distance2 * distance2 ? distance1 : distance2;

	if (radius * radius >= distance * distance)
	{
		const double direction = distance < 0 ? -1.0 : 1.0;
		distance *= direction;
		const double velocity = direction * distance * fZeroPerMuZero * (radius - distance) / (radius * distance);
		atomicAdd(&v[idx], velocity);
	}
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

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec lbb, dvec tfr, ivec cellDim, int numBubbles)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numBubbles; i += gridDim.x * blockDim.x)
	{
		const int cellIdx = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
		cellIndices[i] = cellIdx;
		bubbleIndices[i] = i;
	}
}

__global__ void neighborSearch(int neighborCellNumber,
							   int numValues,
							   int numCells,
							   int numMaxPairs,
							   int *offsets,
							   int *sizes,
							   int *first,
							   int *second,
							   double *r,
							   double intervalX, bool wrapX, double *x,
							   double intervalY, bool wrapY, double *y,
							   double intervalZ, bool wrapZ, double *z)
{
	const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
	const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
	const int cellIdx2 = getNeighborCellIndex(idxVec, dimVec, neighborCellNumber, wrapX, wrapY, wrapZ);

	if (cellIdx2 >= 0)
	{
		const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);
		DEVICE_ASSERT(cellIdx1 < numCells);
		DEVICE_ASSERT(cellIdx2 < numCells);

		if (sizes[cellIdx1] == 0 || sizes[cellIdx2] == 0)
			return;

		// Self comparison only loops the upper triangle of values (n * (n - 1)) / 2 comparisons instead of n^2.
		if (cellIdx1 == cellIdx2)
		{
			const int size = sizes[cellIdx1];
			const int offset = offsets[cellIdx1];
			for (int k = threadIdx.x; k < (size * (size - 1)) / 2; k += blockDim.x)
			{
				int idx1 = size - 2 - (int)floor(sqrt(-8.0 * k + 4 * size * (size - 1) - 7) * 0.5 - 0.5);
				const int idx2 = offset + k + idx1 + 1 - size * (size - 1) / 2 + (size - idx1) * ((size - idx1) - 1) / 2;
				idx1 += offset;

				DEVICE_ASSERT(idx1 < numValues);
				DEVICE_ASSERT(idx2 < numValues);
				DEVICE_ASSERT(idx1 != idx2);

#if (NUM_DIM == 3)
				comparePair(idx1, idx2, r, first, second, intervalX, wrapX, x, intervalY, wrapY, y, intervalZ, wrapZ, z);
#else
				comparePair(idx1, idx2, r, first, second, intervalX, wrapX, x, intervalY, wrapY, y);
#endif
			}
		}
		else // Compare all values of one cell to all values of other cell, resulting in n1 * n2 comparisons.
		{
			const int size1 = sizes[cellIdx1];
			const int size2 = sizes[cellIdx2];
			const int offset1 = offsets[cellIdx1];
			const int offset2 = offsets[cellIdx2];
			for (int k = threadIdx.x; k < size1 * size2; k += blockDim.x)
			{
				const int idx1 = offset1 + k / size2;
				const int idx2 = offset2 + k % size2;

				DEVICE_ASSERT(idx1 < numValues);
				DEVICE_ASSERT(idx2 < numValues);
				DEVICE_ASSERT(idx1 != idx2);

#if (NUM_DIM == 3)
				comparePair(idx1, idx2, r, first, second, intervalX, wrapX, x, intervalY, wrapY, y, intervalZ, wrapZ, z);
#else
				comparePair(idx1, idx2, r, first, second, intervalX, wrapX, x, intervalY, wrapY, y);
#endif
			}
		}
	}
}

__global__ void freeAreaKernel(int numValues, double pi, double *r, double *freeArea, double *freeAreaPerRadius)
{
	const int tid = getGlobalTid();
	if (tid < numValues)
	{
		double area = 2.0 * pi * r[tid];
#if (NUM_DIM == 3)
		area *= 2.0 * r[tid];
#endif
		freeArea[tid] = area - freeArea[tid];
		freeAreaPerRadius[tid] = freeArea[tid] / r[tid];
	}
}

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r, double *freeArea, int numBubbles, double invPi, double kappa, double kParam)
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