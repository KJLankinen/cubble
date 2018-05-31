// -*- C++ -*-

#include "CudaKernelWrapper.h"
#include "Macros.h"
#include "CudaContainer.h"

#include <iostream>
#include <curand.h>

cubble::CudaKernelWrapper::CudaKernelWrapper(std::shared_ptr<BubbleManager> bm)
{
    bubbleManager = bm;
}

cubble::CudaKernelWrapper::~CudaKernelWrapper()
{}

void cubble::CudaKernelWrapper::generateBubblesOnGPU(size_t n,
						     int rngSeed,
						     double avgRad,
						     double stdDevRad,
						     dvec lbb,
						     dvec tfr)
{
    size_t blockSize = n > 1024 ? 1024 : n;
    size_t numBlocks = n > 1024 ? (n % 1024 != 0 ? n / 1024 + 1 : n / 1024) : 1;
    
    CudaContainer<float> x(n);
    CudaContainer<float> y(n);
#if (NUM_DIM == 3)
    CudaContainer<float> z(n);
#endif
    
    CudaContainer<float> r(n);
    CudaContainer<Bubble> b(n);

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
    
    CURAND_CALL(curandGenerateUniform(generator, x.data(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.data(), n));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniform(generator, z.data(), n));
#endif
    CURAND_CALL(curandGenerateNormal(generator, r.data(), n, avgRad, stdDevRad));

    assignDataToBubbles<<<numBlocks, blockSize>>>(x.data(),
						  y.data(),
#if (NUM_DIM == 3)
						  z.data(),
#endif
						  r.data(),
						  b.data(),
						  lbb,
						  tfr);
    
    b.copyDeviceDataToVec(bubbleManager->bubbles);
}

// ----------------------
// Kernel implementations
// ----------------------

__global__
void cubble::assignDataToBubbles(float *x,
				 float *y,
#if (NUM_DIM == 3)
				 float *z,
#endif
				 float *r,
				 Bubble *b,
				 dvec lbb,
				 dvec tfr)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    b[tid].pos.x = lbb.x + (double)x[tid] * tfr.x;
    b[tid].pos.y = lbb.y + (double)y[tid] * tfr.y;
#if (NUM_DIM == 3)
    b[tid].pos.z = lbb.z + (double)z[tid] * tfr.z;
#endif
    b[tid].radius = (double)r[tid];
}
