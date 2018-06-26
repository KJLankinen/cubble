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
	Bubble() {}

	__host__ __device__
	Bubble(int zero) {}
	
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
	void setPosPrev(const dvec &p) { posPrev = p; }
	
	__host__ __device__
	dvec getPosPred() const { return posPred; }
	__host__ __device__
	void setPosPred(const dvec &p) { posPred = p; }
	
	__host__ __device__
	dvec getVel() const { return vel; }
	__host__ __device__
        void setVel(const dvec &v) { vel = v; }
	
	__host__ __device__
	dvec getVelPrev() const { return velPrev; }
	__host__ __device__
        void setVelPrev(const dvec &v) { velPrev = v; }
	
	__host__ __device__
	dvec getVelPred() const { return velPred; }
	__host__ __device__
        void setVelPred(const dvec &v) { velPred = v; }

	__host__ __device__
	dvec getAcc() const { return acc; }
	__host__ __device__
	void setAcc(const dvec &a) { acc = a; }

	fvec getColor() const { return color; }
	__host__ __device__
	void setColor(const fvec &c) { color = c; }
	__host__ __device__
	void setColor(const fvec &&c) { color = c; }
	
	__host__ __device__
	int getCellIndex() const { return cellIndex; }
	__host__ __device__
	void setCellIndex(int i) { cellIndex = i; }
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPos() << ", " << b.getRadius() << ", " << b.getColor();
	    
	    return os;
	}

    private:
	int cellIndex = 0;
	
        double radius = 0;

	dvec pos = dvec(0, 0, 0);
	dvec posPrev = dvec(0, 0, 0);
	dvec posPred = dvec(0, 0, 0);
	
	dvec vel = dvec(0, 0, 0);
	dvec velPrev = dvec(0, 0, 0);
	dvec velPred = dvec(0, 0, 0);

	dvec acc = dvec(0, 0, 0);

	fvec color = fvec(0, 0, 0);
    };
}
