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
	    : devPtr(createDevPtr(n), destroyDevPtr)
	    , hostPtr(createHostPtr(n), destroyHostPtr)
	    , numElements(n)
	{
	    T t(0);
	    for (size_t i = 0; i < numElements; ++i)
		hostPtr[i] = t;
	    
	    this->toDevice();
	}

	CudaContainer(const std::vector<T> &v)
	    : CudaContainer(v.size())
	{
	    for (size_t i = 0; i < v.size(); ++i)
		hostPtr[i] = v[i];
	}
	
	~CudaContainer() {}
	
	T* getDevPtr()
	{
	    return devPtr.get();
	}

	T* getHostPtr()
	{
	    return hostPtr.get();
	}
	
	size_t size()
	{
	    return numElements;
	}
	
	void toHost()
	{
	    CUDA_CALL(cudaMemcpy((void*)hostPtr.get(),
				 (void*)devPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	void toDevice()
	{
	    CUDA_CALL(cudaMemcpy((void*)devPtr.get(),
				 (void*)hostPtr.get(),
				 numElements * sizeof(T),
				 cudaMemcpyHostToDevice));
	}

	void copyVecToHost(const std::vector<T> &v)
	{
	    assert(v.size() == numElements);
	    std::memcpy((void*)hostPtr.get(), (void*)v.data(), numElements * sizeof(T));
	}

        void copyHostDataToVec(std::vector<T> &v)
	{
	    v.clear();
	    v.resize(numElements);
	    std::memcpy((void*)v.data(), (void*)hostPtr.get(), numElements * sizeof(T));
	}

	void copyDeviceDataToVec(std::vector<T> &v)
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
