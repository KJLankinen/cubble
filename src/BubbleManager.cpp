#include "BubbleManager.h"

using namespace cubble;

// Amount of data per bubble:
// 3 vectors of dimension NUM_DIM (position, velocity, acceleration),
// 1 scalar (radius)
const size_t BubbleManager::dataStride = 3 * NUM_DIM + 1;

// Relative locations of data
const size_t BubbleManager::rLoc = 0;
const size_t BubbleManager::pLoc = BubbleManager::rLoc + 1;
const size_t BubbleManager::vLoc = BubbleManager::pLoc + NUM_DIM;
const size_t BubbleManager::aLoc = BubbleManager::vLoc + NUM_DIM;
    
BubbleManager::BubbleManager(size_t numMaxBubbles,
			     int rngSeed,
			     double avgRad,
			     double stdDevRad,
			     double minRad)
    : rngSeed(rngSeed)
    , avgRad(avgRad)
    , stdDevRad(stdDevRad)
    , minRad(minRad)
{
    generator = std::mt19937(rngSeed);
    uniDist = urdd(0, 1);
    normDist = ndd(avgRad, stdDevRad);
    
    data.reserve(numMaxBubbles * dataStride);
}

BubbleManager::~BubbleManager()
{}

size_t BubbleManager::getNumBubbles() const
{
    assert(data.size() % dataStride == 0 && "Data is misaligned.");
    return data.size() / dataStride;
}

double BubbleManager::generateBubble()
{
    BubbleData b;
    for (size_t i = 0; i < NUM_DIM; ++i)
    {
	b.p[i] = uniDist(generator);
        b.v[i] = 0;
        b.a[i] = 0;
    }

    b.r = normDist(generator);
    while (b.r < minRad)
	b.r = normDist(generator);

    // Add the data to the data vector.
    addData(b);
    
    return b.r;
}

void BubbleManager::removeData(size_t i)
{
    auto b = data.begin() + i * dataStride;
    auto e = data.begin() + (i + 1) * dataStride;

    data.erase(b, e);
}

double BubbleManager::getRadius(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + rLoc;
    assert(index < data.size() && "Given index is out of bounds.");
    
    return data[index];
}

void BubbleManager::setRadius(size_t i, double radius)
{
    assert(radius > 0 && "Given radius is negative.");
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + rLoc;
    assert(index < data.size() && "Given index is out of bounds.");
    
    data[index] = radius;
}

dvec BubbleManager::getPosition(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + pLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = data[index + j];
    
    return retVec;
}

void BubbleManager::setPosition(size_t i, dvec position)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + pLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    for (size_t j = 0; j < NUM_DIM; ++j)
        data[index + j] = position[j];
}

dvec BubbleManager::getVelocity(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + vLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = data[index + j];
    
    return retVec;
}

void BubbleManager::setVelocity(size_t i, dvec velocity)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + vLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    for (size_t j = 0; j < NUM_DIM; ++j)
        data[index + j] = velocity[j];
}

dvec BubbleManager::getAcceleration(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + aLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = data[index + j];
    
    return retVec;
}

void BubbleManager::setAcceleration(size_t i, dvec acceleration)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + aLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    for (size_t j = 0; j < NUM_DIM; ++j)
        data[index + j] = acceleration[j];
}

void BubbleManager::addData(const BubbleData &bubbleData)
{
    for (double val : bubbleData.getValues())
	data.push_back(val);
}
