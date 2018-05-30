// -*- C++ -*-

#pragma once

#include "Macros.h"

#include <memory>
#include <array>
#include <iostream>
#include <assert.h>

namespace cubble
{
    template <typename T, size_t N>
    class CudaContainer
    {
    public:
	CudaContainer()
	    : rawPtr(create(), destroy)
	{
	    T t;
	    hostData.fill(t);
	}
	
	~CudaContainer() {}
	
	void fillHostWith(T val)
	{
	    hostData.fill(val);
	}
	
	T* data()
	{
	    return rawPtr.get();
	}
	
	size_t size()
	{
	    return N;
	}
	
	void toHost()
	{
	    CUDA_CALL(cudaMemcpy((void*)hostData.data(),
				 (void*)rawPtr.get(),
				 N * sizeof(T),
				 cudaMemcpyDeviceToHost));
	}
	
	void toDevice()
	{
	    CUDA_CALL(cudaMemcpy((void*)rawPtr.get(),
				 (void*)hostData.data(),
				 N * sizeof(T),
				 cudaMemcpyHostToDevice));
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
	T* create()
	{
	    T *t;
	    CUDA_CALL(cudaMalloc((void**)&t, N * sizeof(T)));
	    return t;
	}

	static void destroy(T *t)
	{
	    CUDA_CALL(cudaFree(t));
	}
	
	std::array<T, N> hostData;
	std::unique_ptr<T[], decltype(&destroy)> rawPtr;
    };
};
