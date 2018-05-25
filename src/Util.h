// -*- C++ -*-
#pragma once

#include <stdexcept>
#include <iostream>
#include <exception>
#include <memory>
#include <array>
#include <cuda.h>

#include "include/json.hpp"

#include "Macros.h"

namespace cubble
{
    inline void handleException(const std::exception_ptr pExc)
    {
	using json = nlohmann::json; 
	try
	{
	    if (pExc)
		std::rethrow_exception(pExc);
	}
	catch (const json::exception &e)
	{
	    std::cout << "Encountered a json parse error."
		      << "\nMake sure the .json file is correct and filenames are correct.\n"
		      << e.what()
		      << std::endl;
	}
	catch (const std::exception &e)
	{
	    std::cout << "Unhandled exception!\n" << e.what() << std::endl;
	    throw e;
	}
    }

    template <typename T, size_t N>
    class CudaWrapper
    {
    public:
	CudaWrapper()
	    : rawPtr(create(), destroy)
	{
	    T t;
	    hostData.fill(t);
	}
	
	~CudaWrapper()
	{}
	
	T* get() { return rawPtr.get(); }
	
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
	    if (i < hostData.size())
		return hostData[i];
	    else
	    {
		std::cerr << "Index i = " << i
			  << " is out of bounds for array of size " << hostData.size() << "."
			  << std::endl;
		return (T)0;
	    }
	}
	
	T& operator[](size_t i)
	{
	    if (i < hostData.size())
		return hostData[i];
	    else
	    {
		std::cerr << "Index i = " << i
			  << " is out of bounds for array of size " << hostData.size() << "."
			  << std::endl;
		return hostData[0];
	    }
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
    
    const double epsilon = 1.0e-10;
}
