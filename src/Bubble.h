// -*- C++ -*-

#pragma once

#include "Vec.h"

#include <iostream>
#include <cuda_runtime.h>

namespace cubble
{
    class Bubble
    {
	// See Macros.h for details of this macro
	CUBBLE_HOST_DEVICE_PROP(int, CellIndex)
	CUBBLE_HOST_DEVICE_PROP(double, Radius)
	CUBBLE_HOST_DEVICE_PROP(dvec, Pos)
	CUBBLE_HOST_DEVICE_PROP(dvec, PosPrev)
	CUBBLE_HOST_DEVICE_PROP(dvec, PosPred)
	CUBBLE_HOST_DEVICE_PROP(dvec, Vel)
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPrev)
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPred)
	
    public:
	__host__ __device__
	Bubble() {}

	__host__ __device__
	Bubble(int zero) {}

	__host__ __device__
	~Bubble() {}
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPos() << ", " << b.getRadius();
	    
	    return os;
	}
    };
}
