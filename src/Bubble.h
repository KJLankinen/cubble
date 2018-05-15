// -*- C++ -*-

#pragma once

#include "Vec.h"

#include <iostream>

namespace cubble
{
    class Bubble
    {
    public:
	Bubble()
	    : uid(newUID++)
	{}
	
	Bubble(vec<double, 3> pos, double r)
	    : Bubble()
	{
	    position = pos;
	    radius = r;
	}
	
	~Bubble() {}

	inline void updatePosition(const vec<double, 3> &newPos)
	{
	    previousPosition = position;
	    setPosition(newPos);
	}   
	inline void setPosition(const vec<double, 3> &newPos) { position = newPos; }
	inline void setPosition(const vec<double, 3> &&newPos) { position = newPos; }
	inline vec<double, 3> getPosition() const { return position; }	
	inline vec<double, 3> getPreviousPosition() const { return previousPosition; }
	
	inline double getRadius() const { return radius; }
	inline void setRadius(double newRad) { radius = newRad; }
	
	inline size_t getUID() const { return uid; }
	
	inline void setCellIndex(size_t i) { cellIndex = i; }
	inline size_t getCellIndex() const { return cellIndex; }
	
	inline void updateVelocity(vec<double, 3> &vel)
	{
	    previousVelocity = velocity;
	    setVelocity(vel);
	}
	inline void updateVelocity(vec<double, 3> &&vel)
	{
	    previousVelocity = velocity;
	    setVelocity(vel);
	}
	inline void setVelocity(const vec<double, 3> &vel) { velocity = vel; }
	inline void setVelocity(const vec<double, 3> &&vel) { velocity = vel; }
	inline vec<double, 3> getVelocity() { return velocity; }
	inline vec<double, 3> getPreviousVelocity() { return previousVelocity; }
	
	inline void addForce(const vec<double, 3> &dF) { force += dF; }
	inline void addForce(const vec<double, 3> &&dF){ force += dF; }
	inline void resetForce()
	{
	    previousForce = force;
	    force.setComponent(0, 0);
	    force.setComponent(0, 1);
	    force.setComponent(0, 2);
	}
	inline vec<double, 3> getForce() { return force; }
	inline vec<double, 3> getPreviousForce() { return previousForce; }
	
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
	
        vec<double, 3> position;
	vec<double, 3> previousPosition;
	vec<double, 3> velocity;
	vec<double, 3> previousVelocity;
	vec<double, 3> force;
	vec<double, 3> previousForce;
    };
}
