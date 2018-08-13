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
	void reserveMemory();
	void resetData(BubbleProperty prop);
	
    private:
	void freeMemory();
	
	void *rawDeviceMemoryPtr = nullptr;
	
	const size_t givenNumBubbles;
	size_t stride = 0;
	size_t numBubbles = 0;
    };
}
