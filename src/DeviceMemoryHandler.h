// -*- C++ -*-

#pragma once

#include <cuda_runtime.h>

namespace cubble
{
    enum BubbleProperty
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

	// Add all temporary data (= data that is guaranteed to be up to date only for one iteration)
	// here, so they can be reset with just one cudaMemset call.
	ERROR,
	ENERGY,
	VOLUME,
	
	NUM_VALUES
    };
    
    class DeviceMemoryHandler
    {
    public:
	DeviceMemoryHandler(size_t numBubbles);
	~DeviceMemoryHandler();
	double *getDataPtr(BubbleProperty prop);
	void swapData();
	void reserveMemory();
        double *getRawPtr();
	double *getRawPtrToTemporaryData();
	void resetData(BubbleProperty prop);
	void resetShortLivedData();
	size_t getNumValuesInMemory() { return stride * BubbleProperty::NUM_VALUES; }
        size_t getMemorySizeInBytes() { return sizeof(double) * getNumValuesInMemory(); }
	size_t getMemoryStride() { return stride; }
	
    private:
	void freeMemory();
	
	void *rawDeviceMemoryPtr = nullptr;
	
	const size_t givenNumBubbles;
	size_t stride = 0;
    };
}
