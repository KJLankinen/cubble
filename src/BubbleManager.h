// -*- C++ -*-

#pragma once

#include "Bubble.h"

#include <vector>

namespace cubble
{
    class BubbleManager
    {
    public:
	BubbleManager();
	~BubbleManager();

	std::vector<Bubble> bubbles;
	std::vector<int> indices;
    };
}
