#include "Kernels.cuh"

namespace cubble
{

__device__ bool dErrorEncountered;
__device__ int dMaxBubblesPerCell;
__device__ int dNumPairs;
__device__ double dTotalFreeArea;
__device__ double dTotalFreeAreaPerRadius;
__device__ double dVolumeMultiplier;
__device__ double dTotalVolume;
__device__ double dInvRho;
__device__ double dTotalArea;
__device__ double dAverageSurfaceAreaIn;

__device__ void logError(bool condition, const char *statement, const char *errMsg)
{
    if (condition == false)
    {
        printf("----------------------------------------------------"
               "\nError encountered"
               "\n(%s) -> %s"
               "\n@thread[%d, %d, %d], @block[%d, %d, %d]"
               "\n----------------------------------------------------\n",
               statement, errMsg,
               threadIdx.x, threadIdx.y, threadIdx.z,
               blockIdx.x, blockIdx.y, blockIdx.z);

        dErrorEncountered = true;
    }
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
__device__ void resetDoubleArrayToValue(double value, int idx, double *array)
{
    array[idx] = value;
}

__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values, double constant)
{
    flags[idx] = values[idx] < constant ? 1 : 0;
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags, double *values, double constant)
{
    flags[idx] = values[idx] > constant ? 1 : 0;
}

__device__ double getWrappedDistance(double x1, double x2, double maxDistance, bool shouldWrap)
{
    const double distance = x1 - x2;
    x2 = distance < -0.5 * maxDistance ? x2 - maxDistance : (distance > 0.5 * maxDistance ? x2 + maxDistance : x2);
    const double distance2 = x1 - x2;

    return shouldWrap ? distance2 : distance;
}

__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x)
{
    const double distance = getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap);
    DEVICE_ASSERT(distance * distance > 0, "Distance is zero!");
    return distance * distance;
}
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, double minDistance, bool shouldWrap, double *x, double *useless)
{
    return getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
}

__global__ void transformPositionsKernel(bool normalize, int numValues, dvec lbb, dvec tfr, double *x, double *y, double *z)
{
    const dvec interval = tfr - lbb;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
    {
        if (normalize)
        {
            x[i] = (x[i] - lbb.x) / interval.x;
            y[i] = (y[i] - lbb.y) / interval.y;
#if (NUM_DIM == 3)
            z[i] = (z[i] - lbb.z) / interval.z;
#endif
        }
        else
        {
            x[i] = interval.x * x[i] + lbb.x;
            y[i] = interval.y * y[i] + lbb.y;
#if (NUM_DIM == 3)
            z[i] = interval.z * z[i] + lbb.z;
#endif
        }
    }
}

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
        printf("Should never end up here!\n");
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

__device__ void wrapAround(int idx, double *coordinate, double minValue, double maxValue, bool *wrappedFlags)
{
    const double interval = maxValue - minValue;
    double value = coordinate[idx];
    value = value < minValue ? value + interval : (value > maxValue ? value - interval : value);
    wrappedFlags[idx] = coordinate[idx] != value ? !wrappedFlags[idx] : wrappedFlags[idx];
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

__device__ void addNeighborVelocity(int idx1, int idx2, double *sumOfVelocities, double *velocity)
{
    atomicAdd(&sumOfVelocities[idx1], velocity[idx2]);
    atomicAdd(&sumOfVelocities[idx2], velocity[idx1]);
}

__device__ void addFlowVelocity(int idx, int *numNeighbors, double *flowVelocity, double *velocity)
{
    velocity[idx] += (numNeighbors[idx] > 0 ? 1.0 / numNeighbors[idx] : 0.0) * flowVelocity[idx];
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
        cellIndices[i] = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
        bubbleIndices[i] = i;
    }
}

__global__ void freeAreaKernel(int numValues, double pi, double *r, double *freeArea, double *freeAreaPerRadius, double *area)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
    {
        double totalArea = 2.0 * pi * r[i];
#if (NUM_DIM == 3)
        totalArea *= 2.0 * r[i];
#endif
        area[i] = totalArea;
        freeArea[i] = totalArea - freeArea[i];
        freeAreaPerRadius[i] = freeArea[i] / r[i];
    }
}

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r, double *freeArea, int numValues, double invPi, double kappa, double kParam)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues; i += gridDim.x * blockDim.x)
    {
        dInvRho = dTotalFreeAreaPerRadius / dTotalFreeArea;
        const double invRadius = 1.0 / r[i];
        double invArea = 0.5 * invPi * invRadius;
#if (NUM_DIM == 3)
        invArea *= 0.5 * invRadius;
#endif
        const double vr = drdt[i] + kappa * dAverageSurfaceAreaIn * numValues / dTotalArea * freeArea[i] * (dInvRho - invRadius);
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

__device__ void adamsBashforth(int idx, double timeStep, double *yNext, double *y, double *f, double *fPrevious)
{
    yNext[idx] = y[idx] + 0.5 * timeStep * (3.0 * f[idx] - fPrevious[idx]);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext, double *y, double *f, double *fNext)
{
    const double error = y[idx] + 0.5 * timeStep * (f[idx] + fNext[idx]) - yNext[idx];
    yNext[idx] += error;

    return error < 0 ? -error : error;
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f)
{
    y[idx] += f[idx] * timeStep;
}

__device__ double calculateDistanceFromStart(int idx, double *x, double *xPrev, double *xStart, bool *wrapped, double interval)
{
    double distance = x[idx] - xStart[idx];
    distance = wrapped[idx] ? (interval + (distance < 0 ? distance : -distance)) : distance;
    return distance * distance;
}

__device__ double calculatePathLength(int idx, double *x, double *xPrev, double *xStart, bool *wrapped, double interval)
{
    const double diff = x[idx] - xPrev[idx];
    return diff * diff;
}

} // namespace cubble