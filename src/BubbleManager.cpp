#include "BubbleManager.h"

#include <iostream>

using namespace cubble;

BubbleManager::BubbleManager(std::shared_ptr<Env> e)
{
    env = e;
}

BubbleManager::~BubbleManager()
{}

void BubbleManager::setBubbles(const std::vector<Bubble> &b)
{
    bubbles = b;
}

void BubbleManager::setBubblesFromDevice(CudaContainer<Bubble> &b)
{
    b.copyDeviceDataToVec(bubbles);
}

void BubbleManager::getBubbles(std::vector<Bubble> &b) const
{
    b = bubbles;
}

void BubbleManager::getBubbles(CudaContainer<Bubble> &b) const
{
    b.copyVecToHost(bubbles);
}

void BubbleManager::setIndicesFromDevice(CudaContainer<int> &i)
{
    i.copyDeviceDataToVec(indices);
}

void BubbleManager::getIndices(std::vector<int> &i) const
{
    i = indices;
}

void BubbleManager::getIndices(CudaContainer<int> &i) const
{
    i.copyVecToHost(indices);
}

void BubbleManager::printIndices() const
{
    for (const auto &it : indices)
	std::cout << it << std::endl;
}

void BubbleManager::setCellsFromDevice(CudaContainer<Cell> &c)
{
    c.copyDeviceDataToVec(cells);
}

void BubbleManager::getCells(std::vector<Cell> &c) const
{
    c = cells;
}

void BubbleManager::getCells(CudaContainer<Cell> &c) const
{
    c.copyVecToHost(cells);
}

Bubble BubbleManager::getBubble(size_t i) const
{
    assert(i < bubbles.size());
    return bubbles[i];
}
