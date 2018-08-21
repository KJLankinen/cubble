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

void cubble::DeviceMemoryHandler::swapData()
{
    cudaMemcpy(static_cast<void*>(rawDeviceMemoryPtr),
	       static_cast<void*>(getRawPtrToTemporaryData()),
	       getPermanentMemorySizeInBytes(),
	       cudaMemcpyDeviceToDevice);
}

void cubble::DeviceMemoryHandler::resetTemporaryData()
{
    cudaMemset(static_cast<void*>(getRawPtrToTemporaryData()), 0, getTemporaryMemorySizeInBytes());
}

double* cubble::DeviceMemoryHandler::getDataPtr(cubble::BubbleProperty prop)
{
    // Note that this function exposes raw device memory. Proceed with caution.
    size_t propIdx = (size_t)prop;
    assert(propIdx < (size_t)BubbleProperty::NUM_VALUES);
    
    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");
    
    dataPtr = getRawPtr();
    dataPtr += propIdx * stride;
    
    assert(dataPtr < getRawPtr() + getNumPermanentValuesInMemory());

    return dataPtr;
}

double* cubble::DeviceMemoryHandler::getDataPtr(TemporaryBubbleProperty prop)
{
    // Note that this function exposes raw device memory. Proceed with caution.
    size_t propIdx = (size_t)prop;
    assert(propIdx < (size_t)BubbleProperty::NUM_VALUES);

    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");

    dataPtr = getRawPtrToTemporaryData();
    dataPtr += propIdx * stride;

    assert(dataPtr < getRawPtr() + getNumPermanentValuesInMemory() + getNumTemporaryValuesInMemory());

    return dataPtr;
}

double *cubble::DeviceMemoryHandler::getDataPtr(AccelerationProperty prop)
{
    // Note that this function exposes raw device memory. Proceed with caution.
    size_t propIdx = (size_t)prop;
    assert(propIdx < (size_t)AccelerationProperty::NUM_VALUES);

    double *dataPtr = nullptr;
    assert(rawDeviceMemoryPtr != nullptr
	   && "Device memory pointer is a nullptr! Can't get a data pointer from the memory handler!");

    dataPtr = getRawPtrToTemporaryData();
    dataPtr += propIdx * stride * neighborStride;

    assert(dataPtr < getRawPtr() + getNumPermanentValuesInMemory() + getNumTemporaryValuesInMemory());

    return dataPtr;
}

double* cubble::DeviceMemoryHandler::getRawPtr()
{
    return static_cast<double*>(rawDeviceMemoryPtr);
}

double* cubble::DeviceMemoryHandler::getRawPtrToTemporaryData()
{
    double *tempPtr = getRawPtr();
    tempPtr += getNumPermanentValuesInMemory();

    return tempPtr;
}

size_t cubble::DeviceMemoryHandler::getNumTemporaryValuesInMemory() const
{
    return (size_t)AccelerationProperty::NUM_VALUES * neighborStride * stride;
}

size_t cubble::DeviceMemoryHandler::getNumBytesOfMemoryFromPropertyToEnd(TemporaryBubbleProperty prop) const
{
    size_t valuesBefore = ((size_t)prop + (size_t)BubbleProperty::NUM_VALUES) * stride;
    
    return (getNumPermanentValuesInMemory() + getNumTemporaryValuesInMemory() - valuesBefore) * sizeof(double);
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
