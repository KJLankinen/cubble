#include "DeviceMemoryHandler.h"

#include <cmath>
#include <assert.h>
#include <iostream>

cubble::DeviceMemoryHandler::DeviceMemoryHandler(size_t numBubbles, size_t neighborStride)
    : givenNumBubbles(numBubbles)
    , neighborStride(neighborStride)
{
    stride = (size_t)std::ceil((float)givenNumBubbles / 32.0f) * 32;
    std::cout << "Number of bubbles given: " << givenNumBubbles << ", leading to stride of " << stride << std::endl;
}

cubble::DeviceMemoryHandler::~DeviceMemoryHandler()
{
    freeMemory();
}

void cubble::DeviceMemoryHandler::reserveMemory()
{
    // This only allocates memory for the 'double' properties of bubbles.
    // 'Auxiliary' (indices &c.) memory isn't allocated yet.
    
    if (rawDeviceMemoryPtr == nullptr)
    {
	size_t numBytes = getPermanentMemorySizeInBytes() + getTemporaryMemorySizeInBytes();
	std::cout << "Allocating " << numBytes << " bytes of device memory for "
		  << givenNumBubbles << " bubbles."
		  << "\nMemory is allocated for the smallest multiplicate of 32 that's >= "
		  << givenNumBubbles << "."
		  << std::endl;
	
	cudaMalloc((void**)&rawDeviceMemoryPtr, numBytes);
	cudaMemset(rawDeviceMemoryPtr, 0, numBytes);
    }
    else
	std::cerr << "Trying to allocate on a pointer that's not a nullptr!"
		  << "\nSomething funky might be going on!"
		  << std::endl;
}

double* cubble::DeviceMemoryHandler::getDataPtr(cubble::BubbleProperty prop)
{
    // Note that this function exposes raw device memory. Proceed with caution.
    size_t propIdx = (size_t)prop;
    assert(propIdx < (size_t)BubbleProperty::NUM_VALUES);
    
    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");
    
    dataPtr = static_cast<double*>(getRawPtrToMemory());
    dataPtr += propIdx * stride;
    
    assert(dataPtr < static_cast<double*>(getRawPtrToMemory()) + getNumPermanentValuesInMemory());

    return dataPtr;
}

double *cubble::DeviceMemoryHandler::getDataPtr(BubblePairProperty prop)
{
    // Note that this function exposes raw device memory. Proceed with caution.
    size_t propIdx = (size_t)prop;
    assert(propIdx < (size_t)BubblePairProperty::NUM_VALUES);

    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");

    dataPtr = getRawPtrToTemporaryData();
    dataPtr += propIdx * stride * neighborStride;

    assert(dataPtr < static_cast<double*>(getRawPtrToMemory())
	   + getNumPermanentValuesInMemory() + getNumTemporaryValuesInMemory());

    return dataPtr;
}

void* cubble::DeviceMemoryHandler::getRawPtrToMemory()
{
    return rawDeviceMemoryPtr;
}

double* cubble::DeviceMemoryHandler::getRawPtrToTemporaryData()
{
    double *tempPtr = static_cast<double*>(getRawPtrToMemory());
    tempPtr += getNumPermanentValuesInMemory();

    return tempPtr;
}

void* cubble::DeviceMemoryHandler::getRawPtrToCubReductionOutputMemory(size_t sizeRequirementInBytes)
{
    if (sizeRequirementInBytes > cubReductionOutputMemorySizeInBytes)
    {
	if (cubReductionOutputPtr)
	{
	    cudaFree(cubReductionOutputPtr);
	    cubReductionOutputPtr = nullptr;
	}

	cudaMalloc((void**)&cubReductionOutputPtr, sizeRequirementInBytes);
	cubReductionOutputMemorySizeInBytes = sizeRequirementInBytes;
    }
    
    return cubReductionOutputPtr;
}

void* cubble::DeviceMemoryHandler::getRawPtrToCubReductionTempMemory(size_t sizeRequirementInBytes)
{
    if (sizeRequirementInBytes > cubReductionTemporaryMemorySizeInBytes)
    {
	if (cubReductionTempPtr)
	{
	    cudaFree(cubReductionTempPtr);
	    cubReductionTempPtr = nullptr;
	}

	cudaMalloc((void**)&cubReductionTempPtr, sizeRequirementInBytes);
	cubReductionTemporaryMemorySizeInBytes = sizeRequirementInBytes;
    }
    
    return cubReductionTempPtr;   
}

size_t cubble::DeviceMemoryHandler::getNumTemporaryValuesInMemory() const
{
    return (size_t)BubblePairProperty::NUM_VALUES * neighborStride * stride;
}

void cubble::DeviceMemoryHandler::freeMemory()
{
    if (rawDeviceMemoryPtr)
    {
	cudaFree(rawDeviceMemoryPtr);
	rawDeviceMemoryPtr = nullptr;
    }
    else
	std::cout << "Device memory pointer is a nullptr, can't free the memory." << std::endl;
    
    if (cubReductionOutputPtr)
    {
	cudaFree(cubReductionOutputPtr);
	cubReductionOutputPtr = nullptr;
    }
    else
	std::cout << "Device memory pointer is a nullptr, can't free the memory." << std::endl;
    
    if (cubReductionTempPtr)
    {
	cudaFree(cubReductionTempPtr);
	cubReductionTempPtr = nullptr;
    }
    else
	std::cout << "Device memory pointer is a nullptr, can't free the memory." << std::endl;
}
