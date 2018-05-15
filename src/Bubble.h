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
	
	Bubble(dvec pos, double r)
	    : Bubble()
	{
	    position = pos;
	    radius = r;
	}
	
	~Bubble() {}

	inline void updatePosition(const dvec &newPos)
	{
	    previousPosition = position;
	    setPosition(newPos);
	}   
	inline void setPosition(const dvec &newPos) { position = newPos; }
	inline void setPosition(const dvec &&newPos) { position = newPos; }
	inline dvec getPosition() const { return position; }	
	inline dvec getPreviousPosition() const { return previousPosition; }
	
	inline double getRadius() const { return radius; }
	inline void setRadius(double newRad) { radius = newRad; }
	
	inline size_t getUID() const { return uid; }
	
	inline void setCellIndex(size_t i) { cellIndex = i; }
	inline size_t getCellIndex() const { return cellIndex; }
	
	inline void updateVelocity(dvec &vel)
	{
	    previousVelocity = velocity;
	    setVelocity(vel);
	}
	inline void updateVelocity(dvec &&vel)
	{
	    previousVelocity = velocity;
	    setVelocity(vel);
	}
	inline void setVelocity(const dvec &vel) { velocity = vel; }
	inline void setVelocity(const dvec &&vel) { velocity = vel; }
	inline dvec getVelocity() { return velocity; }
	inline dvec getPreviousVelocity() { return previousVelocity; }
	
	inline void addForce(const dvec &dF) { force += dF; }
	inline void addForce(const dvec &&dF){ force += dF; }
	inline void resetForce()
	{
	    previousForce = force;
	    force.setComponent(0, 0);
	    force.setComponent(0, 1);
	    force.setComponent(0, 2);
	}
	inline dvec getForce() { return force; }
	inline dvec getPreviousForce() { return previousForce; }
	
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
	
        dvec position;
	dvec previousPosition;
	dvec velocity;
	dvec previousVelocity;
	dvec force;
	dvec previousForce;
    };
}
