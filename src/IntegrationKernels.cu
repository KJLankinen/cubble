#include "IntegrationKernels.cuh"

namespace cubble
{
__device__ void adamsBashforth(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fPrevious)
{
    yNext[idx] = y[idx] + 0.5 * timeStep * (3.0 * f[idx] - fPrevious[idx]);
}

__device__ CubbleFloatType adamsMoulton(int idx, CubbleFloatType timeStep, CubbleFloatType *yNext, CubbleFloatType *y, CubbleFloatType *f, CubbleFloatType *fNext)
{
    const CubbleFloatType yTemp = y[idx] + 0.5 * timeStep * (f[idx] + fNext[idx]);
    CubbleFloatType error = yTemp - yNext[idx];
    error = error < 0 ? -error : error;
    yNext[idx] = yTemp;

    return error;
}

__device__ void eulerIntegrate(int idx, CubbleFloatType timeStep, CubbleFloatType *y, CubbleFloatType *f)
{
    y[idx] += f[idx] * timeStep;
}

}