// -*- C++ -*-

#pragma once

#include "Macros.h"

#include <memory>
#include <vector>
#include <iostream>
#include <assert.h>
#include <cstring>

namespace cubble
{
    template <typename T>
    class CudaContainer
    {
    public:
	CudaContainer(size_t n)
	    : numElements(n)
	    , hostPtr(createHostPtr(n), destroyHostPtr)
	    , devPtr(createDevPtr(n), destroyDevPtr)
	{
	    T t(0);
	    for (size_t i = 0; i < numElements; ++i)
		hostPtr[i] = t;
	    
	    hostToDevice();
	}

	CudaContainer(const std::vector<T> &v)
	    : numElements(v.size())
	    , hostPtr(createHostPtr(v.size()), destroyHostPtr)
	    , devPtr(createDevPtr(v.size()), destroyDevPtr)
	{
	    for (size_t i = 0; i < v.size(); ++i)
		hostPtr[i] = v[i];

	    hostToDevice();
	}
	
	~CudaContainer() {}
	
	T* getDevPtr()
	{
	    return devPtr.get();
	}
	
	size_t size()
	{
	    return numElements;
	}
	
	void deviceToHost()
	{
	    CUDA_CALL(cudaMemcpy((void*)hostPtr.get(),
				 (void*)devPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	void hostToDevice()
	{
	    CUDA_CALL(cudaMemcpy((void*)devPtr.get(),
				 (void*)hostPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyHostToDevice));
	}

        void hostToVec(std::vector<T> &v)
	{
	    v.clear();
	    v.resize(numElements);
	    std::memcpy((void*)v.data(), (void*)hostPtr.get(), numElements * sizeof(T));
	}

	void deviceToVec(std::vector<T> &v)
	{
	    v.clear();
	    v.resize(numElements);
	    CUDA_CALL(cudaMemcpy((void*)v.data(),
				 (void*)devPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	T operator[](size_t i) const
	{
	    assert(i < numElements);
	    return hostPtr[i];
	}
	
	T& operator[](size_t i)
	{
	    assert(i < numElements);
	    return hostPtr[i];
	}

	void operator=(CudaContainer<T> &&o)
	{
	    numElements = o.numElements;
	    hostPtr = std::move(o.hostPtr);
	    devPtr = std::move(o.devPtr);
	}

	void operator=(const CudaContainer<T> &&o)
	{
	    numElements = o.numElements;
	    hostPtr = std::move(o.hostPtr);
	    devPtr = std::move(o.devPtr);
	}
	
    private:
	T* createDevPtr(size_t numElements)
	{
	    T *t;
	    CUDA_CALL(cudaMalloc((void**)&t, numElements * sizeof(T)));
	    return t;
	}

	static void destroyDevPtr(T *t)
	{
	    CUDA_CALL(cudaFree(t));
	}

	
	T* createHostPtr(size_t numElements)
	{
	    T *t = new T[numElements];
	    return t;
	}

	static void destroyHostPtr(T *t)
	{
	    delete[] t;
	}

	size_t numElements = 0;
        
	std::unique_ptr<T[], decltype(&destroyHostPtr)> hostPtr;
	std::unique_ptr<T[], decltype(&destroyDevPtr)> devPtr;
    };
};
