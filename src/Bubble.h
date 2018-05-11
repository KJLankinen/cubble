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

	inline void updatePosition(Vector3<double> newPos)
	{
	    previousPosition = position;
	    setPosition(newPos);
	}   
	inline void setPosition(Vector3<double> newPos) { position = newPos; }
	inline Vector3<double> getPosition() const { return position; }	
	inline Vector3<double> getPreviousPosition() const { return previousPosition; }
	
	inline double getRadius() const { return radius; }
	inline void setRadius(double newRad) { radius = newRad; }
	
	inline size_t getUID() const { return uid; }
	
	inline void setCellIndex(size_t i) { cellIndex = i; }
	inline size_t getCellIndex() const { return cellIndex; }
	
	inline void updateVelocity(Vector3<double> &vel)
	{
	    previousVelocity = velocity;
	    setVelocity(vel);
	}
	inline setVelocity(Vector3<double> &vel) { velocity = vel; }
	inline Vector3<double> getVelocity() { return velocity; }
	inline Vector3<double> getPreviousVelocity() { return previousVelocity; }
	
	inline void addForce(Vector3<double> &dF) { force += dF; }
	inline void resetForce()
	{
	    previousForce = force;
	    force.setX(0);
	    force.setY(0);
	    force.setZ(0);
	}
	inline Vector3<double> getForce() { return force; }
	inline Vector3<double> getPreviousForce() { return perviousForce; }
	
	bool overlapsWith(const Bubble &b) const;
	double getOverlapRadiusSquared(const Bubble &b) const;
	
	friend std::ostream& operator<<(std::ostream &os, const Bubble &b)
	{
	    os << b.getPosition() << ", " << b.getRadius();
	    
	    return os;
	}
	
    private:
	static size_t newUID;

        size_t uid; // <-- Initialized by the constructor to a unique value
	size_t cellIndex = ~0U;
	
        double radius = 0;
	
        Vector3<double> position = Vector3<double>(0, 0, 0);
	Vector3<double> previousPosition = Vector3<double>(0, 0, 0);
	Vector3<double> velocity = Vector3<double>(0, 0, 0);
	Vector3<double> previousVelocity = Vector3<double>(0, 0, 0);
	Vector3<double> force = Vector3<double>(0, 0, 0);
	Vector3<double> previousForce = Vector3<double>(0, 0, 0);
    };
}
