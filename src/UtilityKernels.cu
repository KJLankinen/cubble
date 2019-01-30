#include "UtilityKernels.cuh"

namespace cubble
{
__device__ bool dErrorEncountered;

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
__device__ void resetDoubleArrayToValue(CubbleFloatType value, int idx, CubbleFloatType *array)
{
    array[idx] = value;
}

__device__ void setFlagIfLessThanConstant(int idx, int *flags, CubbleFloatType *values, CubbleFloatType constant)
{
    flags[idx] = values[idx] < constant ? 1 : 0;
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags, CubbleFloatType *values, CubbleFloatType constant)
{
    flags[idx] = values[idx] > constant ? 1 : 0;
}

__device__ CubbleFloatType getWrappedDistance(CubbleFloatType x1, CubbleFloatType x2, CubbleFloatType maxDistance, bool shouldWrap)
{
    const CubbleFloatType distance = x1 - x2;
    x2 = distance < -0.5 * maxDistance ? x2 - maxDistance : (distance > 0.5 * maxDistance ? x2 + maxDistance : x2);
    const CubbleFloatType distance2 = x1 - x2;

    return shouldWrap ? distance2 : distance;
}

__device__ CubbleFloatType getDistanceSquared(int idx1, int idx2, CubbleFloatType maxDistance, bool shouldWrap, CubbleFloatType *x)
{
    const CubbleFloatType distance = getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap);
    DEVICE_ASSERT(distance * distance > 0, "Distance is zero!");
    return distance * distance;
}
__device__ CubbleFloatType getDistanceSquared(int idx1, int idx2, CubbleFloatType maxDistance, CubbleFloatType minDistance, bool shouldWrap, CubbleFloatType *x, CubbleFloatType *useless)
{
    return getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
}

__global__ void transformPositionsKernel(bool normalize, int numValues, fpvec lbb, fpvec tfr, CubbleFloatType *x, CubbleFloatType *y, CubbleFloatType *z)
{
    const fpvec interval = tfr - lbb;
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

} // namespace cubble