// -*- C++ -*-

#pragma once

#include "Bubble.h"

#include <vector>
#include <assert.h>

namespace cubble
{
    class BubbleManager
    {
    public:
	BubbleManager();
	~BubbleManager();

	void setBubbles(const std::vector<Bubble> &b)
	{
	    bubbles = b;
	}

	void getBubbles(std::vector<Bubble> &b)
	{
	    b = bubbles;
	}

	Bubble getBubble(size_t i)
	{
	    assert(i < bubbles.size());
	    return bubbles[i];
	}
	
	std::vector<size_t> cellBegins;
	std::vector<size_t> cellEnds;
    private:
	
	std::vector<Bubble> bubbles;
    };
}
