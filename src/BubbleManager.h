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

	void swapData();
	void updateTemporary();
	
	size_t getNumBubbles() const;

	double generateBubble();
	void removeData(size_t i);

	double getRadius(size_t i, bool useTemporary = false) const;
	void setRadius(size_t i, double radius, bool useTemporary = false);
	
	dvec getPosition(size_t i, bool useTemporary = false) const;
	void updatePosition(size_t i, dvec position);

	dvec getVelocity(size_t i, bool useTemporary = false) const;
	dvec getPrevVelocity(size_t i) const;
	void updateVelocity(size_t i, dvec velocity);

	const static size_t dataStride;
	const static size_t rLoc;
	const static size_t pLoc;
	const static size_t vLoc;
	const static size_t vPrevLoc;
	
    private:
	struct BubbleData
	{
	    double r = 0.0;
	    dvec p;
	    dvec v;
	    dvec vp;

	    const std::vector<double> getValues() const
	    {
		std::vector<double> retVec;
	        retVec.push_back(r);
		retVec.push_back(p.x);
		retVec.push_back(p.y);
#if (NUM_DIM == 3)
		retVec.push_back(p.z);
#endif
		retVec.push_back(v.x);
		retVec.push_back(v.y);
#if (NUM_DIM == 3)
		retVec.push_back(v.z);
#endif
		retVec.push_back(vp.x);
		retVec.push_back(vp.y);
#if (NUM_DIM == 3)
		retVec.push_back(vp.z);
#endif
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
	std::vector<double> temporaryData;
    };
}
