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
						     size_t numBlocksPerDim,
						     int rngSeed,
						     double avgRad,
						     double stdDevRad,
						     dvec lbb,
						     dvec tfr)
{
#if (NUM_DIM == 3)
    size_t blockSize = std::ceil(n / (float)(numBlocksPerDim
					     * numBlocksPerDim
					     * numBlocksPerDim));
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim, numBlocksPerDim);
    size_t recommendedNumCells = (size_t)(std::ceil(std::cbrt(n / 1024.0f) / 8)) * 8;
#else
    size_t blockSize = std::ceil(n / (float)(numBlocksPerDim * numBlocksPerDim));
    dim3 gridSize = dim3(numBlocksPerDim, numBlocksPerDim);
    size_t recommendedNumCells = (size_t)(std::ceil(std::sqrt(n / 1024.0f) / 8)) * 8;
#endif

    if (blockSize > 1024)
    {
	std::cerr << "Too many bubbles to simulate w.r.t. the number of blocks."
		  << "\nAmount of threads per block " << blockSize
		  << " exceeds the maximum, which is 1024."
		  << "\nIncrease the number of blocks (cells) per dimension."
		  << "\nRecommended number of cells with given number of bubbles: "
		  << recommendedNumCells
		  << std::endl;
	
	std::exit(EXIT_FAILURE);
    }

    std::cout << "Grid size: " << gridSize.x << ", " << gridSize.y << ", " << gridSize.z
	      << ", block size: " << blockSize
	      << "\nRecommended number of cells with given number of bubbles: "
	      << recommendedNumCells
	      << std::endl;
    
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
    
    CURAND_CALL(curandGenerateUniform(generator, x.getDevicePtr(), n));
    CURAND_CALL(curandGenerateUniform(generator, y.getDevicePtr(), n));
#if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniform(generator, z.getDevicePtr(), n));
#endif
    CURAND_CALL(curandGenerateNormal(generator, r.getDevicePtr(), n, avgRad, stdDevRad));

    assignDataToBubbles<<<gridSize, blockSize>>>(x.getDevicePtr(),
						  y.getDevicePtr(),
#if (NUM_DIM == 3)
						  z.getDevicePtr(),
#endif
						  r.getDevicePtr(),
						  b.getDevicePtr(),
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
#if (NUM_DIM == 3)
    // 3D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.z * (gridDim.y * gridDim.x)
		  + blockIdx.y * gridDim.x + blockIdx.x)
	* blockDim.x + threadIdx.x;
#else
    // 2D grid of blocks with 1D blocks of threads.
    size_t tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
#endif
    b[tid].pos.x = lbb.x + (double)x[tid] * tfr.x;
    b[tid].pos.y = lbb.y + (double)y[tid] * tfr.y;
#if (NUM_DIM == 3)
    b[tid].pos.z = lbb.z + (double)z[tid] * tfr.z;
#endif
    b[tid].radius = (double)r[tid];
}
