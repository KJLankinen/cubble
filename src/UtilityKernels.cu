#include "UtilityKernels.cuh"

namespace cubble
{
__device__ bool dErrorEncountered;

__device__ void logError(bool condition, const char *statement, const char *errMsg)
{
    if (condition == false)
    {
        printf("----------------------------------------------------\n");
        printf("Error encountered.\n(%s) -> %s\n", statement, errMsg);
        printf("@thread[%d, %d, %d], @block[%d, %d, %d]\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        printf("----------------------------------------------------\n");

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

} // namespace cubble