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
	CUBBLE_HOST_DEVICE_PROP(int, CellIndex, 0)
	CUBBLE_HOST_DEVICE_PROP(double, Radius, 0)
	CUBBLE_HOST_DEVICE_PROP(dvec, Pos, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, PosPrev, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, PosPred, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, Vel, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPrev, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPred, dvec(0, 0, 0))
	
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
