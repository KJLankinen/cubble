// -*- C++ -*-

#pragma once

#include <cuda_runtime.h>

namespace cubble
{
    class CudaKernelsWrapper
    {
    public:
        CudaKernelsWrapper() {}
	~CudaKernelsWrapper() {}
	
	void testFunctionWrapper();
    };

    // Kernel has to be defined outside class
    __global__
    void testFunction(float *a, float *b);
};
