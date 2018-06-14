#include "BubbleManager.h"

#include <iostream>
#include <math.h>

using namespace cubble;

BubbleManager::BubbleManager(std::shared_ptr<Env> e)
{
    env = e;
}

BubbleManager::~BubbleManager()
{}

double BubbleManager::getVolumeOfBubbles() const
{
    double volume = 0;
    
    for (const auto &bubble : bubbles)
    {
	double radius = bubble.getRadius();
	
#if NUM_DIM == 3
	volume += radius * radius * radius;
#else
	volume += radius * radius;
#endif
    }
    volume *= M_PI;
    
#if NUM_DIM == 3
    volume *=  1.3333333333333333333333333;
#endif

    return volume;
}

void BubbleManager::setBubbles(const std::vector<Bubble> &b)
{
    bubbles = b;
}

void BubbleManager::setBubblesFromDevice(CudaContainer<Bubble> &b)
{
    b.deviceToVec(bubbles);
}

void BubbleManager::getBubbles(std::vector<Bubble> &b) const
{
    b = bubbles;
}

void BubbleManager::getBubbles(CudaContainer<Bubble> &b) const
{
    b = CudaContainer<Bubble>(bubbles);
}

void BubbleManager::setIndicesFromDevice(CudaContainer<int> &i)
{
    i.deviceToVec(indices);
}

void BubbleManager::getIndices(std::vector<int> &i) const
{
    i = indices;
}

void BubbleManager::getIndices(CudaContainer<int> &i) const
{
    i = CudaContainer<int>(indices);
}

void BubbleManager::setCellsFromDevice(CudaContainer<Cell> &c)
{
    c.deviceToVec(cells);
}

void BubbleManager::getCells(std::vector<Cell> &c) const
{
    c = cells;
}

void BubbleManager::getCells(CudaContainer<Cell> &c) const
{
    c = CudaContainer<Cell>(cells);
}

Bubble BubbleManager::getBubble(size_t i) const
{
    assert(i < bubbles.size());
    return bubbles[i];
}
