#include "DeviceMemoryHandler.h"

#include <cmath>
#include <assert.h>
#include <iostream>

cubble::DeviceMemoryHandler::DeviceMemoryHandler(size_t numBubbles)
    : givenNumBubbles(numBubbles)
    , numBubbles(numBubbles)
{
    stride = (size_t)std::ceil((float)numBubbles / 32.0f) * 32;
    std::cout << "Number of bubbles given: " << givenNumBubbles << ", leading to stride of " << stride << std::endl;
}

cubble::DeviceMemoryHandler::~DeviceMemoryHandler()
{
    freeMemory();
}

double* cubble::DeviceMemoryHandler::getDataPtr(cubble::BubbleProperty prop)
{
    // ASSUMPTION: Only properties of type 'double' are used.
    // Note that this function exposes raw device memory. Proceed with caution.

    assert(prop < cubble::BubbleProperty::NUM_VALUES);
    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");
    
    dataPtr = static_cast<double*>(rawDeviceMemoryPtr);
    dataPtr += prop * stride;

    return dataPtr;
}

void cubble::DeviceMemoryHandler::reserveMemory()
{
    // This only allocates memory for the 'double' properties of bubbles.
    // 'Auxiliary' (indices &c.) memory isn't allocated yet.
    
    if (rawDeviceMemoryPtr == nullptr)
    {
	size_t numBytes = stride * sizeof(double) * BubbleProperty::NUM_VALUES;
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

void cubble::DeviceMemoryHandler::resetData(cubble::BubbleProperty prop)
{
    assert(prop < cubble::BubbleProperty::NUM_VALUES);
    
    size_t numBytes = stride * sizeof(double);
    cudaMemset(getDataPtr(prop), 0, numBytes);
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
}
