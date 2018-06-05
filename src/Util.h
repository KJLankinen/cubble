// -*- C++ -*-
#pragma once

#include <iostream>
#include <stdexcept>
#include <exception>
#include <sstream>

#include <cuda.h>
#include <curand.h>

#ifndef __CUDACC__
  #include "include/json.hpp"
#endif

#include "Macros.h"

namespace cubble
{
    const double epsilon = 1.0e-10;
    
    inline void handleException(const std::exception_ptr pExc)
    {
#ifndef __CUDACC__
	using json = nlohmann::json;
#endif
	try
	{
	    if (pExc)
		std::rethrow_exception(pExc);
	}
#ifndef __CUDACC__
	catch (const json::exception &e)
	{
	    std::cout << "Encountered a json parse error."
		      << "\nMake sure the .json file is correct and filenames are correct.\n"
		      << e.what()
		      << std::endl;
	}
#endif
	catch (const std::exception &e)
	{
	    std::cout << "---------------------------------------------------------------"
		      << "\nUnhandled exception: " << e.what()
		      << "\n---------------------------------------------------------------"
		      << std::endl;
	    throw e;
	}
    }
    
    inline void cudaAssert(cudaError_t result,
			   const char *file,
			   int line,
			   bool abort = true)
    {
	std::stringstream ss;
	if (result != cudaSuccess)
	{
	    ss << "Cuda error encountered at " << file << ":" << line << ": ";
	    ss << cudaGetErrorString(result) << "\n";
	    
	    if (abort)
	    {
		ss << "Throwing...\n";
		throw std::runtime_error(ss.str());
	    }
	    else
		std::cerr << ss.str() << std::endl;
	}
    }
    
    inline void curandAssert(curandStatus_t result,
			     const char *file,
			     int line,
			     bool abort = true)
    {
	std::stringstream ss;
	if (result != CURAND_STATUS_SUCCESS)
	{
	    ss << "Curand error encountered at " << file << ":" << line << "\n";
	    
	    if (abort)
	    {
		ss << "Throwing...\n";
		throw std::runtime_error(ss.str());
	    }
	    else
		std::cerr << ss.str() << std::endl;
	}
    }

    inline void compareSizeOfDynamicSharedMemoryToDeviceLimit(int bytes, bool abort = true)
    {
	int value = 0;
	int device = 0;
	cudaDeviceAttr attr = cudaDevAttrMaxSharedMemoryPerBlock;

	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaDeviceGetAttribute(&value, attr, device));
	
	if (bytes > value)
	{
	    std::stringstream ss;
	    ss << "Requested size of dynamically allocated shared memory exceeds";
	    ss << " the limitation of the current device.";
	    ss << "\nRequested size: " << bytes;
	    ss << "\nDevice limit: " << value;

	    if (abort)
	    {
		ss << "\nThrowing...";
		throw std::runtime_error(ss.str());
	    }
	    else
		std::cerr << ss.str() << std::endl;
	}
    }
}
