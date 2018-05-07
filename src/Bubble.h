// -*- C++ -*-

#pragma once

#include "Vector3.h"

#include <iostream>

namespace cubble
{
    class Bubble
    {
    public:
	Bubble(Vector3<double> pos, double r)
	    : position(pos)
	    , radius(r)
	{}
	Bubble() {}
	Bubble(Bubble &&o)
	    : position(o.position)
	    , radius(o.radius)
	{}
	~Bubble() {}

	inline Vector3<double> getPosition() const { return position; }
	inline void setPosition(Vector3<double> newPos) { position = newPos; }
	inline double getRadius() const { return radius; }
	inline void setRadius(double newRad) { radius = newRad; }
	bool overlapsWith(const Bubble &b) const;
	double getOverlapRadiusSquared(const Bubble &b) const;

	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPosition() << ", " << b.getRadius();
	    
	    return os;
	}
	
    private:
        Vector3<double> position = Vector3<double>(0, 0, 0);
        double radius = 0;
    };
}
