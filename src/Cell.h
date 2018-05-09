// -*- C++ -*-

#pragma once

#include "Bubble.h"

#include <map>

namespace cubble
{
    class Cell
    {
    public:
	Cell() {}
	~Cell() {}

	void addNeighborRef(Cell *cell);
	void addBubbleRef(Bubble *bubble);
	void removeBubbleRef(size_t key);
	void getBubbleRefsAsVector(std::vector<Bubble*> &bubbles) const;
    private:
	std::vector<Cell*> neighbors;
	std::map<size_t, Bubble*> bubbleMap;
    };
}
