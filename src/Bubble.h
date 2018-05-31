// -*- C++ -*-

#pragma once

#include "Vec.h"

#include <iostream>
#include <cuda_runtime.h>

namespace cubble
{
    class Bubble
    {
    public:
	__host__ __device__
	Bubble()
	{}
	
	__host__ __device__
	Bubble(double r, dvec pos)
	    : radius(r)
	    , pos(pos)
	{}

	__host__ __device__
	~Bubble() {}
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.pos << ", " << b.radius;
	    
	    return os;
	}

        double radius = 0;

	dvec pos;
	dvec posPrev;
	dvec posPred;
	
	dvec vel;
	dvec velPrev;
	dvec velPred;

	dvec acc;
    };
}
