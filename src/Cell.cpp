#include "Cell.h"

using namespace cubble;

void Cell::addNeighborRef(Cell *cell)
{
    if (cell)
	neighbors.push_back(cell);
    else
	std::cerr << "Cell to add was a nullptr!" << std::endl;
}

void Cell::addBubbleRef(Bubble *bubble)
{
    if (bubble)
	bubbleMap[bubble->getUID()] = bubble;
    else
	std::cerr << "Bubble to add was a nullptr!" << std::endl;
}

void Cell::removeBubbleRef(size_t key)
{
    bubbleMap.erase(key);
}

void Cell::getBubbleRefsAsVector(std::vector<Bubble*> &bubbles) const
{
    for (auto pair : bubbleMap)
	bubbles.push_back(pair.second);
}
