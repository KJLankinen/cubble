// -*- C++ -*-

#include "BubbleKernels.cuh"
#include "UtilityKernels.cuh"

namespace cubble
{
__device__ int dMaxBubblesPerCell;
__device__ int dNumPairs;
__device__ CubbleFloatType dTotalFreeArea;
__device__ CubbleFloatType dTotalFreeAreaPerRadius;
__device__ CubbleFloatType dVolumeMultiplier;
__device__ CubbleFloatType dTotalVolume;
__device__ CubbleFloatType dInvRho;
__device__ CubbleFloatType dTotalArea;
__device__ CubbleFloatType dAverageSurfaceAreaIn;

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

__device__ CubbleFloatType getWrappedCoordinate(CubbleFloatType val1, CubbleFloatType val2, CubbleFloatType multiplier)
{
	CubbleFloatType difference = val1 - val2;
	val2 = difference < -0.5 * multiplier ? val2 - multiplier : (difference > 0.5 * multiplier ? val2 + multiplier : val2);
	val2 = val1 - val2;

	return val2;
}

__device__ int getCellIdxFromPos(CubbleFloatType x, CubbleFloatType y, CubbleFloatType z, fpvec lbb, fpvec tfr, ivec cellDim)
{
	const fpvec interval = tfr - lbb;
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

__device__ void wrapAround(int idx, CubbleFloatType *coordinate, CubbleFloatType minValue, CubbleFloatType maxValue)
{
	const CubbleFloatType interval = maxValue - minValue;
	CubbleFloatType value = coordinate[idx];
	value = value < minValue ? value + interval : (value > maxValue ? value - interval : value);
	coordinate[idx] = value;
}

__device__ void addVelocity(int idx1, int idx2, CubbleFloatType multiplier, CubbleFloatType maxDistance, CubbleFloatType minDistance, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v)
{
	const CubbleFloatType velocity = getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap) * multiplier;
	atomicAdd(&v[idx1], velocity);
	atomicAdd(&v[idx2], -velocity);
}

__device__ void forceFromWalls(int idx, CubbleFloatType fZeroPerMuZero, CubbleFloatType *r,
							   CubbleFloatType interval, CubbleFloatType zeroPoint, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *v)
{
	if (shouldWrap)
		return;

	const CubbleFloatType radius = r[idx];
	const CubbleFloatType distance1 = x[idx] - zeroPoint;
	const CubbleFloatType distance2 = x[idx] - (interval + zeroPoint);
	CubbleFloatType distance = distance1 * distance1 < distance2 * distance2 ? distance1 : distance2;

	if (radius * radius >= distance * distance)
	{
		const CubbleFloatType direction = distance < 0 ? -1.0 : 1.0;
		distance *= direction;
		const CubbleFloatType velocity = direction * distance * fZeroPerMuZero * (radius - distance) / (radius * distance);
		atomicAdd(&v[idx], velocity);
	}
}

__global__ void calculateVolumes(CubbleFloatType *r, CubbleFloatType *volumes, int numValues, CubbleFloatType pi)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		CubbleFloatType radius = r[i];
		CubbleFloatType volume = radius * radius * pi;
#if (NUM_DIM == 3)
		volume *= radius * 1.33333333333333333333333333;
#endif

		volumes[i] = volume;
	}
}

__global__ void assignDataToBubbles(CubbleFloatType *x, CubbleFloatType *y, CubbleFloatType *z,
									CubbleFloatType *xPrd, CubbleFloatType *yPrd, CubbleFloatType *zPrd,
									CubbleFloatType *r,
									CubbleFloatType *w,
									int *aboveMinRadFlags,
									ivec bubblesPerDim,
									fpvec tfr,
									fpvec lbb,
									CubbleFloatType avgRad,
									CubbleFloatType minRad,
									CubbleFloatType pi,
									int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		fpvec pos(0, 0, 0);
		pos.x = (i % bubblesPerDim.x) / (CubbleFloatType)bubblesPerDim.x;
		pos.y = ((i / bubblesPerDim.x) % bubblesPerDim.y) / (CubbleFloatType)bubblesPerDim.y;

		fpvec randomOffset(x[i], y[i], 0);
#if (NUM_DIM == 3)
		randomOffset.z = z[i];
		pos.z = (i / (bubblesPerDim.x * bubblesPerDim.y)) / (CubbleFloatType)bubblesPerDim.z;
#endif
		pos *= tfr - lbb;
		randomOffset = fpvec::normalize(randomOffset) * avgRad * w[i];
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

__global__ void assignBubblesToCells(CubbleFloatType *x, CubbleFloatType *y, CubbleFloatType *z, int *cellIndices, int *bubbleIndices, fpvec lbb, fpvec tfr, ivec cellDim, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		const int cellIdx = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
		cellIndices[i] = cellIdx;
		bubbleIndices[i] = i;
	}
}

__global__ void freeAreaKernel(int numValues, CubbleFloatType pi, CubbleFloatType *r, CubbleFloatType *relativeFreeArea, CubbleFloatType *relativeFreeAreaPerRadius, CubbleFloatType *area)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		CubbleFloatType totalArea = 2.0 * pi * r[i];
#if (NUM_DIM == 3)
		totalArea *= 2.0 * r[i];
#endif
		area[i] = totalArea;
		relativeFreeArea[i] = totalArea - relativeFreeArea[i];
		relativeFreeAreaPerRadius[i] = relativeFreeArea[i] / r[i];
	}
}

__global__ void finalRadiusChangeRateKernel(CubbleFloatType *drdt, CubbleFloatType *r, CubbleFloatType *relativeFreeArea, int numValues, CubbleFloatType invPi, CubbleFloatType kappa, CubbleFloatType kParam)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		dInvRho = dTotalFreeAreaPerRadius / dTotalFreeArea;
		const CubbleFloatType invRadius = 1.0 / r[i];
		CubbleFloatType invArea = 0.5 * invPi * invRadius;
#if (NUM_DIM == 3)
		invArea *= 0.5 * invRadius;
#endif
		const CubbleFloatType vr = drdt[i] + kappa * dAverageSurfaceAreaIn * numValues / dTotalArea * relativeFreeArea[i] * (dInvRho - invRadius);
		drdt[i] = kParam * invArea * vr;
	}
}

__global__ void addVolume(CubbleFloatType *r, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		CubbleFloatType multiplier = dVolumeMultiplier / dTotalVolume;
		multiplier += 1.0;

#if (NUM_DIM == 3)
		multiplier = cbrt(multiplier);
#else
		multiplier = sqrt(multiplier);
#endif
		r[i] = r[i] * multiplier;
	}
}

__global__ void calculateRedistributedGasVolume(CubbleFloatType *volume, CubbleFloatType *r, int *aboveMinRadFlags, CubbleFloatType pi, int numValues)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
	{
		const CubbleFloatType radius = r[i];
		CubbleFloatType vol = pi * radius * radius;
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
