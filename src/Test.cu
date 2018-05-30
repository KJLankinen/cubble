// -*- C++ -*-

#include "Test.h"

#include <vector>
#include <numeric>
#include <iostream>

#define CUDA_CALL(x)							\
    do									\
    {									\
	cudaError_t result = x;						\
	if (result != cudaSuccess)					\
	{								\
	    std::cerr << "Error at " << __FILE__ << ":" << __LINE__;	\
	    std::cerr << cudaGetErrorName(result) << "\n"		\
		      << cudaGetErrorString(result) << std::endl;	\
	}								\
    }									\
    while(0)

__global__
void cubble::testFunction(float *a, float *b)
{
    int tid = threadIdx.x;
    printf("Asd asd asd asd from thread %d", tid);
    b[tid] = a[tid] * a[tid];
}

void cubble::Test::testFunctionWrapper()
{
    const size_t n = 1024;
    std::vector<float> a(n);
    std::vector<float> b;
    b.resize(a.size());
    std::iota(a.begin(), a.end(), 0);
    /*
    for (auto it : a)
	std::cout << it << " ";
    
    std::cout << std::endl;
    */
    float *d_a, *d_b;
    CUDA_CALL(cudaMalloc((void**)&d_a, n * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_b, n * sizeof(float)));

    CUDA_CALL(cudaMemcpy((void*)d_a, (void*)a.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Before" << std::endl;
    cubble::testFunction<<<1, n>>>(d_a, d_b);
    CUDA_CALL(cudaMemcpy((void*)b.data(), (void*)d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "After" << std::endl;

    /*
    for (auto it : b)
	std::cout << it << " ";
    
    std::cout << std::endl;
    */
    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
}
