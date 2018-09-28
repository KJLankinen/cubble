#include "IntegrationKernels.cuh"

namespace cubble
{
__device__ void adamsBashforth(int idx, double timeStep, double *yNext, double *y, double *f, double *fPrevious)
{
    yNext[idx] = y[idx] + 0.5 * timeStep * (3.0 * f[idx] - fPrevious[idx]);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext, double *y, double *f, double *fNext)
{
    const double yTemp = y[idx] + 0.5 * timeStep * (f[idx] + fNext[idx]);
    double error = yTemp - yNext[idx];
    error = error < 0 ? -error : error;
    yNext[idx] = yTemp;

    return error;
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f)
{
    y[idx] += f[idx] * timeStep;
}

}