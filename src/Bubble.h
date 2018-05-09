// -*- C++ -*-

#pragma once

#include "Vector3.h"

#include <iostream>

namespace cubble
{
    class Bubble
    {
    public:
	Bubble()
	    : uid(newUID++)
	{}
	
	Bubble(Vector3<double> pos, double r)
	    : Bubble()
	{
	    position = pos;
	    radius = r;
	}
	
	~Bubble() {}

	inline Vector3<double> getPosition() const { return position; }
	inline void setPosition(Vector3<double> newPos) { position = newPos; }
	inline double getRadius() const { return radius; }
	inline void setRadius(double newRad) { radius = newRad; }
	inline size_t getUID() const { return uid; }
	inline void setCellIndex(size_t i) { cellIndex = i; }
	inline size_t getCellIndex() const { return cellIndex; }
	bool overlapsWith(const Bubble &b) const;
	double getOverlapRadiusSquared(const Bubble &b) const;
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPosition() << ", " << b.getRadius();
	    
	    return os;
	}
	
    private:
	static size_t newUID;
        Vector3<double> position = Vector3<double>(0, 0, 0);
        double radius = 0;
	size_t cellIndex = ~0U;
        size_t uid;
    };
}
