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
	CUBBLE_HOST_DEVICE_PROP(double, RadiusPred, 0)
	CUBBLE_HOST_DEVICE_PROP(double, RadiusChangeRate, 0)
	CUBBLE_HOST_DEVICE_PROP(double, RadiusChangeRatePrev, 0)
	CUBBLE_HOST_DEVICE_PROP(double, RadiusChangeRatePred, 0)
	CUBBLE_HOST_DEVICE_PROP(dvec, Pos, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, PosPred, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, Vel, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPrev, dvec(0, 0, 0))
	CUBBLE_HOST_DEVICE_PROP(dvec, VelPred, dvec(0, 0, 0))
	
    public:
	__host__ __device__
	Bubble() {}

	__host__ __device__
	Bubble(int zero) {}

	Bubble(const Bubble &o)
	{
	    *this = o;
	}

	__host__ __device__
	~Bubble() {}
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPos() << ", " << b.getRadius();
	    
	    return os;
	}

	__host__ __device__
	void operator=(Bubble &o)
	{
	    if (&o != this)
	    {
		CellIndex = o.CellIndex;
		Radius = o.Radius;
		RadiusPred = o.RadiusPred;
		RadiusChangeRate = o.RadiusChangeRate;
		RadiusChangeRatePrev = o.RadiusChangeRatePrev;
		RadiusChangeRatePred = o.RadiusChangeRatePred;
		Pos = o.Pos;
		PosPred = o.PosPred;
		Vel = o.Vel;
		VelPrev = o.VelPrev;
		VelPred = o.VelPred;
	    }
	}
	
	__host__ __device__
	void operator=(const Bubble &o)
	{
	    if (&o != this)
	    {
		CellIndex = o.CellIndex;
		Radius = o.Radius;
		RadiusPred = o.RadiusPred;
		RadiusChangeRate = o.RadiusChangeRate;
		RadiusChangeRatePrev = o.RadiusChangeRatePrev;
		RadiusChangeRatePred = o.RadiusChangeRatePred;
		Pos = o.Pos;
		PosPred = o.PosPred;
		Vel = o.Vel;
		VelPrev = o.VelPrev;
		VelPred = o.VelPred;
	    }
	}

	__host__ __device__
	void operator=(Bubble &&o)
	{
	    CellIndex = o.CellIndex;
	    Radius = o.Radius;
	    RadiusPred = o.RadiusPred;
	    RadiusChangeRate = o.RadiusChangeRate;
	    RadiusChangeRatePrev = o.RadiusChangeRatePrev;
	    RadiusChangeRatePred = o.RadiusChangeRatePred;
	    Pos = o.Pos;
	    PosPred = o.PosPred;
	    Vel = o.Vel;
	    VelPrev = o.VelPrev;
	    VelPred = o.VelPred;
	}

	__host__ __device__
	void operator=(const Bubble &&o)
	{
	    CellIndex = o.CellIndex;
	    Radius = o.Radius;
	    RadiusPred = o.RadiusPred;
	    RadiusChangeRate = o.RadiusChangeRate;
	    RadiusChangeRatePrev = o.RadiusChangeRatePrev;
	    RadiusChangeRatePred = o.RadiusChangeRatePred;
	    Pos = o.Pos;
	    PosPred = o.PosPred;
	    Vel = o.Vel;
	    VelPrev = o.VelPrev;
	    VelPred = o.VelPred;
	}
    };
    
    static_assert(sizeof(Bubble) % 4 == 0, "Size must be a multiple of 4.");
}
