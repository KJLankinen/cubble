// -*- C++ -*-

#pragma once

#include "Macros.h"

#include <memory>
#include <vector>
#include <iostream>
#include <assert.h>
#include <algorithm>

namespace cubble
{
    template <typename T>
    class CudaContainer
    {
    public:
	CudaContainer(size_t n)
	    : rawPtr(create(n), destroy)
	    , numElements(n)
	{
	    T t;
	    hostData.resize(n, t);
	}

	CudaContainer(const std::vector<T> &v)
	    : CudaContainer(v.size())
	{
	    hostData = v;
	}
	
	~CudaContainer() {}
	
	void fillHostWith(T val)
	{
	    std::fill(hostData.begin(), hostData.begin() + numElements, val);
	}
	
	T* getDevicePtr()
	{
	    return rawPtr.get();
	}
	
	size_t size()
	{
	    return numElements;
	}
	
	void toHost()
	{
	    hostData.clear();
	    hostData.resize(numElements);
	    CUDA_CALL(cudaMemcpy((void*)hostData.data(),
				 (void*)rawPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	void toDevice()
	{
	    CUDA_CALL(cudaMemcpy((void*)rawPtr.get(),
				 (void*)hostData.data(),
				 numElements * sizeof(T),
				 cudaMemcpyHostToDevice));
	}

        void copyHostDataToVec(std::vector<T> &v)
	{
	    v.clear();
	    v = std::vector<T>(hostData.begin(), hostData.begin() + numElements);
	}

	void copyDeviceDataToVec(std::vector<T> &v)
	{
	    // Need to use resize instead of reserve because copying straigth
	    // to internal memory doesn't update size.
	    v.clear();
	    v.resize(numElements);
	    CUDA_CALL(cudaMemcpy((void*)v.data(),
				 (void*)rawPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	T operator[](size_t i) const
	{
	    assert(i < hostData.size());
	    return hostData[i];
	}
	
	T& operator[](size_t i)
	{
	    assert(i < hostData.size());
	    return hostData[i];
	}
	
    private:
	T* create(size_t numElements)
	{
	    T *t;
	    CUDA_CALL(cudaMalloc((void**)&t, numElements * sizeof(T)));
	    return t;
	}

	static void destroy(T *t)
	{
	    CUDA_CALL(cudaFree(t));
	}

	size_t numElements = 0;
	std::vector<T> hostData;
	std::unique_ptr<T[], decltype(&destroy)> rawPtr;
    };
};
