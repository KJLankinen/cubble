// -*- C++ -*-

#pragma once

#include "Macros.h"

#include <memory>
#include <vector>
#include <iostream>
#include <assert.h>
#include <cstring>

#define DEFAULT_CAPACITY 8

namespace cubble
{
    template <typename T>
    class CudaContainer
    {
    public:
	CudaContainer(size_t n)
	    : capacity(n)
	    , size(n)
	    , hostPtr(createHostPtr(n), destroyHostPtr)
	    , devPtr(createDevPtr(n), destroyDevPtr)
	{
	    T t(0);
	    for (size_t i = 0; i < size; ++i)
		hostPtr[i] = t;
	    
	    hostToDevice();
	}

	CudaContainer()
	    : capacity(DEFAULT_CAPACITY)
	    , size(0)
	    , hostPtr(createHostPtr(DEFAULT_CAPACITY), destroyHostPtr)
	    , devPtr(createDevPtr(DEFAULT_CAPACITY), destroyDevPtr)
	{}
	
	CudaContainer(const std::vector<T> &v)
	    : capacity(v.size())
	    , size(v.size())
	    , hostPtr(createHostPtr(v.size()), destroyHostPtr)
	    , devPtr(createDevPtr(v.size()), destroyDevPtr)
	{
	    for (size_t i = 0; i < size; ++i)
		hostPtr[i] = v[i];

	    hostToDevice();
	}
	
	~CudaContainer() {}
	
	T* getDevPtr() const
	{
	    return devPtr.get();
	}
	
	size_t getSize() const
	{
	    return size;
	}

	size_t getCapacity() const
	{
	    return capacity;
	}
	
	void deviceToHost()
	{
	    CUDA_CALL(cudaMemcpy((void*)hostPtr.get(),
				 (void*)devPtr.get(),
				 size * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	void hostToDevice()
	{
	    CUDA_CALL(cudaMemcpy((void*)devPtr.get(),
				 (void*)hostPtr.get(),
				 size * sizeof(T),
				 cudaMemcpyHostToDevice));
	}

        void hostToVec(std::vector<T> &v) const
	{
	    v.clear();
	    v.resize(size);
	    std::memcpy((void*)v.data(), (void*)hostPtr.get(), size * sizeof(T));
	}

	void deviceToVec(std::vector<T> &v) const
	{
	    v.clear();
	    v.resize(size);
	    CUDA_CALL(cudaMemcpy((void*)v.data(),
				 (void*)devPtr.get(),
				 size * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	T operator[](size_t i) const
	{
	    assert(i < size);
	    return hostPtr[i];
	}
	
	T& operator[](size_t i)
	{
	    assert(i < size);
	    return hostPtr[i];
	}

	void operator=(CudaContainer<T> &&o)
	{
	    capacity = o.capacity;
	    size = o.size;
	    hostPtr = std::move(o.hostPtr);
	    devPtr = std::move(o.devPtr);
	}

	void operator=(const CudaContainer<T> &&o)
	{
	    capacity = o.capacity;
	    size = o.size;
	    hostPtr = std::move(o.hostPtr);
	    devPtr = std::move(o.devPtr);
	}
	
    private:
	T* createDevPtr(size_t capacity)
	{
	    T *t;
	    CUDA_CALL(cudaMalloc((void**)&t, capacity * sizeof(T)));
	    return t;
	}

	static void destroyDevPtr(T *t)
	{
	    CUDA_CALL(cudaFree(t));
	}

	
	T* createHostPtr(size_t capacity)
	{
	    T *t = new T[capacity];
	    return t;
	}

	static void destroyHostPtr(T *t)
	{
	    delete[] t;
	}

	size_t capacity = 0;
	size_t size = 0;
        
	std::unique_ptr<T[], decltype(&destroyHostPtr)> hostPtr;
	std::unique_ptr<T[], decltype(&destroyDevPtr)> devPtr;
    };
};
