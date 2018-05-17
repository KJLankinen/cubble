// -*- C++ -*-

#pragma once

#include <vector>
#include <assert.h>

#include "Vec.h"

typedef std::uniform_real_distribution<double> urdd;
typedef std::normal_distribution<double> ndd;

namespace cubble
{
    class BubbleManager
    {
    public:
	BubbleManager(size_t numMaxBubbles,
		      int rngSeed,
		      double avgRad,
		      double stdDevRad,
		      double minRad);
	
	~BubbleManager();

	size_t getNumBubbles() const;

	double generateBubble();
	void removeData(size_t i);

	double getRadius(size_t i) const;
	void setRadius(size_t i, double radius);
	
	dvec getPosition(size_t i) const;
	void setPosition(size_t i, dvec position);

	dvec getVelocity(size_t i) const;
	void setVelocity(size_t i, dvec velocity);

	dvec getAcceleration(size_t i) const;
	void setAcceleration(size_t i, dvec acceleration);

	const static size_t dataStride;
	const static size_t rLoc;
	const static size_t pLoc;
	const static size_t vLoc;
	const static size_t aLoc;
	
    private:
	struct BubbleData
	{
	    double r = 0.0;
	    dvec p;
	    dvec v;
	    dvec a;

	    const std::vector<double> getValues() const
	    {
		std::vector<double> retVec;
	        retVec.push_back(r);

		for (size_t i = 0; i < NUM_DIM; ++i)
		    retVec.push_back(p[i]);
		
		for (size_t i = 0; i < NUM_DIM; ++i)
		    retVec.push_back(v[i]);
		
		for (size_t i = 0; i < NUM_DIM; ++i)
		    retVec.push_back(a[i]);

		return retVec;
	    }
	};
	
	void addData(const BubbleData &bubbleData);
	
	urdd uniDist;
	ndd normDist;
	std::mt19937 generator;

	const int rngSeed;
	
	const double avgRad;
	const double stdDevRad;
	const double minRad;
	
	std::vector<double> data;
    };
}
