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

    enum class AccelerationProperty
    {
	X,
	Y,
	Z,
	R,
	E,

	NUM_VALUES
    };
    
    class DeviceMemoryHandler
    {
    public:
	DeviceMemoryHandler(size_t numBubbles, size_t neighborStride);
	~DeviceMemoryHandler();
	
	void reserveMemory();
	void swapData();
	void resetTemporaryData();
	
	double *getDataPtr(BubbleProperty prop);
	double *getDataPtr(TemporaryBubbleProperty prop);
	double *getDataPtr(AccelerationProperty prop);
        double *getRawPtr();
	double *getRawPtrToTemporaryData();
	
	size_t getNumPermanentValuesInMemory() const { return stride * (size_t)BubbleProperty::NUM_VALUES; }
	size_t getNumTemporaryValuesInMemory() const;
        size_t getPermanentMemorySizeInBytes() const { return sizeof(double) * getNumPermanentValuesInMemory(); }
	size_t getTemporaryMemorySizeInBytes() const { return sizeof(double) * getNumTemporaryValuesInMemory(); }
	size_t getMemoryStride() const { return stride; }
	size_t getNumBytesOfMemoryFromPropertyToEnd(TemporaryBubbleProperty prop) const;
	
    private:
	void freeMemory();
	
	void *rawDeviceMemoryPtr = nullptr;
	
	const size_t givenNumBubbles;
	const size_t neighborStride;
	size_t stride = 0;
    };
}
