// -*- C++ -*-
#pragma once

#include <iostream>
#include <stdexcept>
#include <exception>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "json.hpp"

#include "Macros.h"

namespace cubble
{

struct ExecutionPolicy
{
	ExecutionPolicy(dim3 gridSize, dim3 blockSize, uint32_t sharedMemBytes, cudaStream_t stream)
		: gridSize(gridSize), blockSize(blockSize), sharedMemBytes(sharedMemBytes), stream(stream)
	{
	}

	ExecutionPolicy()
		: ExecutionPolicy(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0)
	{
	}

	ExecutionPolicy(uint32_t numThreadsPerBlock, uint32_t numTotalThreads)
	{
		blockSize = dim3(numThreadsPerBlock, 1, 1);
		gridSize = dim3((uint32_t)std::ceil(numTotalThreads / (float)numThreadsPerBlock), 1, 1);
		sharedMemBytes = 0;
		stream = 0;
	}

	ExecutionPolicy(uint32_t numThreadsPerBlock, uint32_t numTotalThreads, uint32_t bytes, cudaStream_t s)
	{
		blockSize = dim3(numThreadsPerBlock, 1, 1);
		gridSize = dim3((uint32_t)std::ceil(numTotalThreads / (float)numThreadsPerBlock), 1, 1);
		sharedMemBytes = bytes;
		stream = s;
	}

	dim3 gridSize;
	dim3 blockSize;
	uint32_t sharedMemBytes = 0;
	cudaStream_t stream = 0;
};

const double CUBBLE_EPSILON = 1.0e-10;
#if NUM_DIM == 3
const int CUBBLE_NUM_NEIGHBORS = 13;
#else
const int CUBBLE_NUM_NEIGHBORS = 4;
#endif

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
		std::cout << "\n----------Unhandled exception----------\n"
				  << e.what()
				  << "\n---------------------------------------\n"
				  << std::endl;
		throw e;
	}
}

inline void getFormattedCudaErrorString(cudaError_t result, const char *callStr, const char *file, int line, std::basic_ostream<char> &outStream)
{
	outStream << "Cuda error \"" << cudaGetErrorName(result) << "\" encountered at "
			  << file
			  << ":" << line
			  << ":" << callStr
			  << "\n"
			  << cudaGetErrorString(result)
			  << std::endl;
}

inline bool cudaCallAndLog(cudaError_t result, const char *callStr, const char *file, int line) noexcept
{
	if (result != cudaSuccess)
	{
		getFormattedCudaErrorString(result, callStr, file, line, std::cerr);
		return false;
	}

	return true;
}

inline void cudaCallAndThrow(cudaError_t result, const char *callStr, const char *file, int line) noexcept
{
	if (result != cudaSuccess)
	{
		std::stringstream ss;
		getFormattedCudaErrorString(result, callStr, file, line, ss);
		throw std::runtime_error(ss.str());
	}
}

inline bool curandCallAndLog(curandStatus_t result, const char *callStr, const char *file, int line) noexcept
{
	if (result != CURAND_STATUS_SUCCESS)
	{
		std::cerr << "Curand error " << result << " encountered at "
				  << file
				  << ":" << line
				  << ":" << callStr
				  << std::endl;
		return false;
	}

	return true;
}

inline int getCurrentDeviceAttrVal(cudaDeviceAttr attr)
{
#ifndef NDEBUG
	int value = 0;
	int device = 0;

	CUDA_ASSERT(cudaDeviceSynchronize());
	CUDA_ASSERT(cudaGetDevice(&device));
	CUDA_ASSERT(cudaDeviceGetAttribute(&value, attr, device));

	return value;
#else
	return -1;
#endif
}

inline void assertMemBelowLimit(const char *kernelStr, const char *file, int line, int bytes, bool abort = true)
{
#ifndef NDEBUG
	int value = getCurrentDeviceAttrVal(cudaDevAttrMaxSharedMemoryPerBlock);

	if (bytes > value)
	{
		std::stringstream ss;
		ss << "Requested size of dynamically allocated shared memory exceeds"
		   << " the limitation of the current device."
		   << "\nError location: '" << kernelStr << "' @" << file << ":" << line << "."
		   << "\nRequested size: " << bytes
		   << "\nDevice limit: " << value;

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

inline void assertBlockSizeBelowLimit(const char *kernelStr, const char *file, int line, dim3 blockSize, bool abort = true)
{
#ifndef NDEBUG
	dim3 temp;
	temp.x = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimX);
	temp.y = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimY);
	temp.z = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimZ);

	if (temp.x < blockSize.x || temp.y < blockSize.y || temp.z < blockSize.z)
	{
		std::stringstream ss;
		ss << "Block size exceeds the limitation of the current device"
		   << " in at least one dimension."
		   << "\nError location: '" << kernelStr << "' @" << file << ":" << line << "."
		   << "\nBlock size: (" << blockSize.x
		   << ", " << blockSize.y
		   << ", " << blockSize.z << ")"
		   << "\nDevice limit: (" << temp.x
		   << ", " << temp.y
		   << ", " << temp.z
		   << ")";

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

inline void assertGridSizeBelowLimit(const char *kernelStr, const char *file, int line, dim3 gridSize, bool abort = true)
{
#ifndef NDEBUG
	dim3 temp;
	temp.x = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimX);
	temp.y = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimY);
	temp.z = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimZ);

	if (temp.x < gridSize.x || temp.y < gridSize.y || temp.z < gridSize.z)
	{
		std::stringstream ss;
		ss << "Grid size exceeds the limitation of the current device"
		   << " in at least one dimension."
		   << "\nError location: '" << kernelStr << "' @" << file << ":" << line << "."
		   << "\nGrid size: (" << gridSize.x
		   << ", " << gridSize.y
		   << ", " << gridSize.z << ")"
		   << "\nDevice limit: (" << temp.x
		   << ", " << temp.y
		   << ", " << temp.z
		   << ")";

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

	CUDA_ASSERT(cudaDeviceSynchronize());
	CUDA_ASSERT(cudaGetDevice(&device));
	CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));

	std::cout << "\n----------Properties of current device----------"
			  << "\n\n\tGeneral"
			  << "\n\t-------"
			  << "\n\tName: " << prop.name
			  << "\n\tCompute capability: " << prop.major << "." << prop.minor
			  << "\n\n\tMemory"
			  << "\n\t------"
			  << "\n\tTotal global memory (bytes): " << prop.totalGlobalMem
			  << "\n\tShared memory per block (bytes): " << prop.sharedMemPerBlock
			  << "\n\tTotal constant memory (bytes): " << prop.totalConstMem
			  << "\n\tMaximum number of registers per block: " << prop.regsPerBlock
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
} // namespace cubble
