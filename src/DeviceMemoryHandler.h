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
	    
	ENERGY,
	ERROR,
	VOLUME,
	FREE_AREA,
	
	NUM_VALUES
    };

    enum class BubblePairProperty
    {
	ACCELERATION_X,
	ACCELERATION_Y,
	ACCELERATION_Z,
	ACCELERATION_R,
	ENERGY,
	OVERLAP_AREA,

	NUM_VALUES
    };
    
    class DeviceMemoryHandler
    {
    public:
	DeviceMemoryHandler(size_t numBubbles, size_t neighborStride);
	~DeviceMemoryHandler();
	
	void reserveMemory();
	
	double* getDataPtr(BubbleProperty prop);
	double* getDataPtr(BubblePairProperty prop);
	
        void* getRawPtrToMemory();

	void* getRawPtrToCubReductionOutputMemory(size_t sizeRequirementInBytes);
	void* getRawPtrToCubReductionTempMemory(size_t sizeRequirementInBytes);
	
	size_t getNumPermanentValuesInMemory() const { return stride * (size_t)BubbleProperty::NUM_VALUES; }
	size_t getNumTemporaryValuesInMemory() const;

	size_t getPermanentMemorySizeInBytes() const { return sizeof(double) * getNumPermanentValuesInMemory(); }
	size_t getTemporaryMemorySizeInBytes() const { return sizeof(double) * getNumTemporaryValuesInMemory(); }

	size_t getMemoryStride() const { return stride; }
	
    private:
        double* getRawPtrToTemporaryData();
	void freeMemory();
	
	void *rawDeviceMemoryPtr = nullptr;
	void *cubReductionOutputPtr = nullptr;
	void *cubReductionTempPtr = nullptr;
	
	const size_t givenNumBubbles;
	const size_t neighborStride;
	size_t stride = 0;
	size_t cubReductionOutputMemorySizeInBytes = 0;
	size_t cubReductionTemporaryMemorySizeInBytes = 0;
    };
}
