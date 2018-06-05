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
	    std::cout << "\n----------Unhandled exception----------\n"
		      << e.what()
		      << "\n---------------------------------------\n"
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

    inline int getCurrentDeviceAttrVal(cudaDeviceAttr attr)
    {
#ifndef NDEBUG
	int value = 0;
	int device = 0;
	
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaDeviceGetAttribute(&value, attr, device));

	return value;
#else
	return -1;
#endif
    }
    
    inline void assertMemBelowLimit(int bytes, bool abort = true)
    {
#ifndef NDEBUG
	int value = getCurrentDeviceAttrVal(cudaDevAttrMaxSharedMemoryPerBlock);
	
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
#endif
    }

    inline void printRelevantInfoOfCurrentDevice()
    {
	cudaDeviceProp prop;
	int device = 0;

	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&prop, device));

	std::cout << "\n----------Properties of current device----------"
		  << "\n\n\tGeneral"
		  << "\n\t-------"
		  << "\n\tName: " << prop.name
		  << "\n\tCompute capability: " << prop.major << "." << prop.minor
		  << "\n\n\tMemory"
		  << "\n\t------"
		  << "\n\tTotal global memory (bytes): " << prop.totalGlobalMem
		  << "\n\tShared memory per block (bytes): " << prop.sharedMemPerBlock
		  << "\n\tTotal constant memory (bytes):" << prop.totalConstMem
		  << "\n\n\tWarp, threads, blocks, grid"
		  << "\n\t---------------------------"
		  << "\n\tWarp size: " << prop.warpSize
		  << "\n\tMaximum number of threads per block: " << prop.maxThreadsPerBlock
		  << "\n\tMaximum block size: (" << prop.maxThreadsDim[0]
		  << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")"
		  << "\n\tMaximum grid size: (" << prop.maxGridSize[0]
		  << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")"
		  << "\n\tMultiprocessor count: " << prop.multiProcessorCount
		  << "\n"
		  << "\nIf you want more info, see " << __FILE__ << ":" << __LINE__
		  << "\nand 'Device Management' section of the CUDA Runtime API docs."
		  << "\n------------------------------------------------"
		  << "\n"
		  << std::endl;
    }
}
