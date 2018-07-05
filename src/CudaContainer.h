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
	    , dataPtr(createDataPtr(n), destroyDataPtr)
	{
	    T t(0);
	    for (size_t i = 0; i < size; ++i)
		dataPtr[i] = t;
	}

	CudaContainer()
	    : capacity(DEFAULT_CAPACITY)
	    , size(0)
	    , dataPtr(createDataPtr(DEFAULT_CAPACITY), destroyDataPtr)
	{}
	
	CudaContainer(const std::vector<T> &v)
	    : capacity(v.size())
	    , size(v.size())
	    , dataPtr(createDataPtr(v.size()), destroyDataPtr)
	{
	    for (size_t i = 0; i < size; ++i)
		dataPtr[i] = v[i];
	}
	
	~CudaContainer() {}
	
	T* getDataPtr() const
	{
	    return dataPtr.get();
	}
	
	size_t getSize() const
	{
	    return size;
	}

	size_t getCapacity() const
	{
	    return capacity;
	}

	void dataToVec(std::vector<T> &v) const
	{
	    v.clear();
	    v.resize(size);
	    std::memcpy(v.data(), dataPtr.get(), size);
	}

	T operator[](size_t i) const
	{
	    assert(i < size);
	    return dataPtr[i];
	}
	
	T& operator[](size_t i)
	{
	    assert(i < size);
	    return dataPtr[i];
	}

	void operator=(CudaContainer<T> &&o)
	{
	    capacity = o.capacity;
	    size = o.size;
	    dataPtr = std::move(o.dataPtr);
	}

	void operator=(const CudaContainer<T> &&o)
	{
	    capacity = o.capacity;
	    size = o.size;
	    dataPtr = std::move(o.dataPtr);
	}
	
    private:
	T* createDataPtr(size_t capacity)
	{
	    T *t;
	    CUDA_CALL(cudaMallocManaged((void**)&t, capacity * sizeof(T)));
	    CUDA_CALL(cudaDeviceSynchronize());
	    return t;
	}

	static void destroyDataPtr(T *t)
	{
	    CUDA_CALL(cudaDeviceSynchronize());
	    CUDA_CALL(cudaFree(t));
	}

	size_t capacity = 0;
	size_t size = 0;
        
	std::unique_ptr<T[], decltype(&destroyDataPtr)> dataPtr;
    };
};
