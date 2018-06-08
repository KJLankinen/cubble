#include "BubbleManager.h"

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

void BubbleManager::setOffsets(const std::vector<int> &o)
{
    offsets = o;
}

void BubbleManager::setOffsetsFromDevice(CudaContainer<int> &o)
{
    o.copyDeviceDataToVec(offsets);
}

void BubbleManager::getOffsets(std::vector<int> &o) const
{
    o = offsets;
}

Bubble BubbleManager::getBubble(size_t i) const
{
    assert(i < bubbles.size());
    return bubbles[i];
}
