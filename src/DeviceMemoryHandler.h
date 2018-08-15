// -*- C++ -*-

#pragma once

#include <cuda_runtime.h>

namespace cubble
{
    enum class BubbleProperty
    {
	X,
	Y,
	Z,
	R,

	X_PRD,
	Y_PRD,
	Z_PRD,
	R_PRD,

	DXDT,
	DYDT,
	DZDT,
	DRDT,

	DXDT_PRD,
	DYDT_PRD,
	DZDT_PRD,
	DRDT_PRD,

	DXDT_OLD,
	DYDT_OLD,
	DZDT_OLD,
	DRDT_OLD,
	
	NUM_VALUES
    };

    // Memory for these aren't allocated per se.
    // They're used only temporarily and thus are saved
    // in the secondary (temporary) half of the memory pool.
    enum class TemporaryBubbleProperty
    {
	ENERGY,
	ERROR,
	VOLUME,
	REDUCTION_OUTPUT,
	REDUCTION_TEMP,
	
	NUM_VALUES
    };
    
    class DeviceMemoryHandler
    {
    public:
	DeviceMemoryHandler(size_t numBubbles);
	~DeviceMemoryHandler();
	
	void reserveMemory();
	void swapData();
	void resetTemporaryData();
	
	double *getDataPtr(BubbleProperty prop);
	double *getDataPtr(TemporaryBubbleProperty prop);
        double *getRawPtr();
	double *getRawPtrToTemporaryData();
	
	size_t getNumValuesInMemory() { return stride * (size_t)BubbleProperty::NUM_VALUES; }
        size_t getMemorySizeInBytes() { return sizeof(double) * getNumValuesInMemory(); }
	size_t getMemoryStride() { return stride; }
	size_t getNumBytesOfMemoryFromPropertyToEnd(TemporaryBubbleProperty prop);
	
    private:
	void freeMemory();
	
	void *rawDeviceMemoryPtr = nullptr;
	
	const size_t givenNumBubbles;
	size_t stride = 0;
    };
}
