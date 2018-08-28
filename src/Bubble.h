// -*- C++ -*-

#pragma once

#include "Vec.h"
#include <iostream>

namespace cubble
{
    // This used to be a more complicated class, but now it's just a convenience wrapper around some data
    // and stream output operator.
    class Bubble
    {
	CUBBLE_HOST_DEVICE_PROP(double, Radius, 0)
	CUBBLE_HOST_DEVICE_PROP(dvec, Pos, dvec(0, 0, 0))
	
    public:
	__host__ __device__
	Bubble() {}

	__host__ __device__
	~Bubble() {}
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPos() << ", " << b.getRadius();
	    
	    return os;
	}
    };
}
