// -*- C++ -*-

#include "CudaKernelsWrapper.h"
#include "Macros.h"
#include "CudaContainer.h"

#include <vector>
#include <numeric>
#include <iostream>

__global__
void cubble::testFunction(float *a, float *b)
{
    int tid = threadIdx.x;
    b[tid] = a[tid] * a[tid];
}

void cubble::CudaKernelsWrapper::testFunctionWrapper()
{
    const size_t n = 1024;
    cubble::CudaContainer<float, n> a;
    cubble::CudaContainer<float, n> b;

    for (size_t i = 0; i < a.size(); ++i)
    {
	a[i] = i;
	std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    
    a.toDevice();

    cubble::testFunction<<<1, n>>>(a.data(), b.data());
    
    b.toHost();

    for (size_t i = 0; i < b.size(); ++i)
	std::cout << b[i] << " ";
    std::cout << std::endl;
}
