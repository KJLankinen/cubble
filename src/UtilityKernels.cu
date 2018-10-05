#include "UtilityKernels.cuh"

namespace cubble
{
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

__device__ double getWrappedDistance(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x)
{
    double val1 = x[idx1];
    double val2 = x[idx2];
    double distance = val1 - val2;

    if (shouldWrap)
    {
        val2 = distance < -0.5 * maxDistance ? val2 - maxDistance : (distance > 0.5 * maxDistance ? val2 + maxDistance : val2);
        distance = val1 - val2;
    }

    return distance;
}

__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x)
{
    const double distance = getWrappedDistance(idx1, idx2, maxDistance, shouldWrap, x);
    if (!(distance * distance > 0))
        printf("%d, %d, %f, %d, %f, %f, %f\n", idx1, idx2, maxDistance, shouldWrap, x[idx1], x[idx2], distance);
    DEVICE_ASSERT(distance * distance > 0);
    return distance * distance;
}
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance, bool shouldWrap, double *x, double *useless)
{
    return getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
}

} // namespace cubble