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
__device__ double dInvRho;
__device__ double dTotalArea;
__device__ double dAverageSurfaceAreaIn;

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
	if (idxVec.x < 0 || idxVec.x >= dim.x)
		return -1;
#endif

#if (PBC_Y == 1)
	idxVec.y += dim.y;
	idxVec.y %= dim.y;
#else
	if (idxVec.y < 0 || idxVec.y >= dim.y)
		return -1;
#endif

#if (PBC_Z == 1)
	idxVec.z += dim.z;
	idxVec.z %= dim.z;
#else
	if (idxVec.z < 0 || idxVec.z >= dim.z)
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
#if (NUM_DIM == 3)
	const int zid = floor(cellDim.z * (z - lbb.z) / interval.z);
#else
	const int zid = 0;
#endif

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
#if (NUM_DIM == 3)
	idxVec.z = idx / (cellDim.x * cellDim.y);
#endif

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

__global__ void calculateVolumes(double *r, double *volumes, int numValues, double pi)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		double radius = r[i];
		double volume = radius * radius * pi;
#if (NUM_DIM == 3)
		volume *= radius * 1.33333333333333333333333333;
#endif

		volumes[i] = volume;
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
									double pi,
									int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		dvec pos(0, 0, 0);
		pos.x = (i % bubblesPerDim.x) / (double)bubblesPerDim.x;
		pos.y = ((i / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

		dvec randomOffset(x[i], y[i], 0);
#if (NUM_DIM == 3)
		randomOffset.z = z[i];
		pos.z = (i / (bubblesPerDim.x * bubblesPerDim.y)) / (double)bubblesPerDim.z;
#endif
		pos *= tfr - lbb;
		randomOffset = dvec::normalize(randomOffset) * avgRad * w[i];
		pos += randomOffset;

		x[i] = pos.x;
		y[i] = pos.y;
		z[i] = pos.z;

		xPrd[i] = pos.x;
		yPrd[i] = pos.y;
		zPrd[i] = pos.z;

		wrapAround(i,
				   x, lbb.x, tfr.x,
				   y, lbb.y, tfr.y,
#if (NUM_DIM == 3)
				   z, lbb.z, tfr.z,
				   zPrd, lbb.z, tfr.z,
#endif
				   xPrd, lbb.x, tfr.x,
				   yPrd, lbb.y, tfr.y);

		r[i] = r[i] > 0 ? r[i] : -r[i];
		w[i] = 2.0 * pi * r[i] / numValues;

#if (NUM_DIM == 3)
		w[i] *= 2.0 * r[i];
#endif

		setFlagIfGreaterThanConstant(i, aboveMinRadFlags, r, minRad);
	}
}

__global__ void assignBubblesToCells(double *x, double *y, double *z, int *cellIndices, int *bubbleIndices, dvec lbb, dvec tfr, ivec cellDim, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		const int cellIdx = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
		cellIndices[i] = cellIdx;
		bubbleIndices[i] = i;
	}
}

__global__ void freeAreaKernel(int numValues, double pi, double *r, double *relativeFreeArea, double *relativeFreeAreaPerRadius, double *area)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		double totalArea = 2.0 * pi * r[i];
#if (NUM_DIM == 3)
		totalArea *= 2.0 * r[i];
#endif
		area[i] = totalArea;
		relativeFreeArea[i] = (totalArea - relativeFreeArea[i]) / area[i];
		relativeFreeAreaPerRadius[i] = relativeFreeArea[i] / r[i];
	}
}

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r, double *relativeFreeArea, int numValues, double invPi, double kappa, double kParam)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		dInvRho = dTotalFreeAreaPerRadius / dTotalFreeArea;
		const double invRadius = 1.0 / r[i];
		double invArea = 0.5 * invPi * invRadius;
#if (NUM_DIM == 3)
		invArea *= 0.5 * invRadius;
#endif
		const double vr = drdt[i] + kappa * dAverageSurfaceAreaIn * relativeFreeArea[i] * (dInvRho - invRadius);
		drdt[i] = kParam * invArea * vr;
	}
}

__global__ void addVolume(double *r, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		double multiplier = dVolumeMultiplier / dTotalVolume;
		multiplier += 1.0;

#if (NUM_DIM == 3)
		multiplier = cbrt(multiplier);
#else
		multiplier = sqrt(multiplier);
#endif
		r[i] = r[i] * multiplier;
	}
}

__global__ void calculateRedistributedGasVolume(double *volume, double *r, int *aboveMinRadFlags, double pi, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		const double radius = r[i];
		double vol = pi * radius * radius;
#if (NUM_DIM == 3)
		vol *= 1.333333333333333333333333 * radius;
#endif

		if (aboveMinRadFlags[i] == 0)
		{
			atomicAdd(&dVolumeMultiplier, vol);
			volume[i] = 0;
		}
		else
			volume[i] = vol;
	}
}
} // namespace cubble
