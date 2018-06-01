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

	__host__ __device__
	double getRadius() const { return radius; }
	__host__ __device__
        void setRadius(double r) { radius = r; }
	
	__host__ __device__
	dvec getPos() const { return pos; }
	__host__ __device__
	void setPos(const dvec &p) { pos = p; }
	
	__host__ __device__
	dvec getPosPrev() const { return posPrev; }
	__host__ __device__
	dvec getPosPred() const { return posPred; }
	
	__host__ __device__
	dvec getVel() const { return vel; }
	__host__ __device__
        void setVel(const dvec &v) { vel = v; }
	
	__host__ __device__
	dvec getVelPrev() const { return velPrev; }
	__host__ __device__
	dvec getVelPred() const { return velPred; }

	__host__ __device__
	dvec getAcc() { return acc; }
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPos() << ", " << b.getRadius();
	    
	    return os;
	}

    private:
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
