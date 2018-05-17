#include "BubbleManager.h"

#include <algorithm>

using namespace cubble;

// Amount of data per bubble:
// 5 vectors of dimension NUM_DIM (position, velocity, acceleration),
// 1 scalar (radius)
const size_t BubbleManager::dataStride = 5 * NUM_DIM + 1;

// Relative locations of data
const size_t BubbleManager::rLoc = 0;
const size_t BubbleManager::pLoc = BubbleManager::rLoc + 1;
const size_t BubbleManager::vLoc = BubbleManager::pLoc + NUM_DIM;
const size_t BubbleManager::vPrevLoc = BubbleManager::vLoc + NUM_DIM;
const size_t BubbleManager::aLoc = BubbleManager::vPrevLoc + NUM_DIM;
const size_t BubbleManager::aPrevLoc = BubbleManager::aLoc + NUM_DIM;
    
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

void BubbleManager::swapData()
{
    // Make sure temporary is up to date.
    if (data.size() != temporaryData.size())
	updateTemporary();
    
    std::swap(temporaryData, data);
}

void BubbleManager::updateTemporary()
{
    // This is an expensive function, since we're copying the whole data vector.
    // Call sparingly and only when needed, e.g. after values have been added or removed
    // from data vector.

    temporaryData = data;
}

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

double BubbleManager::getRadius(size_t i, bool useTemporary) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + rLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    const std::vector<double> &dataRef = useTemporary ? temporaryData : data;
    
    return dataRef[index];
}

void BubbleManager::setRadius(size_t i, double radius, bool useTemporary)
{
    assert(radius > 0 && "Given radius is negative.");
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + rLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    std::vector<double> &dataRef = useTemporary ? temporaryData : data;
    dataRef[index] = radius;
}

dvec BubbleManager::getPosition(size_t i, bool useTemporary) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + pLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    const std::vector<double> &dataRef = useTemporary ? temporaryData : data;
    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = dataRef[index + j];
    
    return retVec;
}

void BubbleManager::updatePosition(size_t i, dvec position)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + pLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        temporaryData[index + j] = position[j];
}

dvec BubbleManager::getVelocity(size_t i, bool useTemporary) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + vLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    const std::vector<double> &dataRef = useTemporary ? temporaryData : data;
    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = dataRef[index + j];
    
    return retVec;
}

dvec BubbleManager::getPrevVelocity(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + vPrevLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = data[index + j];
    
    return retVec;
}

void BubbleManager::updateVelocity(size_t i, dvec velocity)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + vLoc;
    size_t prevIndex = i * dataStride + vPrevLoc;
    assert(index < data.size() && "Given index is out of bounds.");
    assert(prevIndex < data.size() && "Given index is out of bounds.");

    for (size_t j = 0; j < NUM_DIM; ++j)
    {
	// Put the new data to the old, then swap old and current
	// so current contains new and old contains current.
	temporaryData[prevIndex + j] = velocity[j];
	std::swap(temporaryData[prevIndex + j], temporaryData[index + j]);
    }
}

dvec BubbleManager::getAcceleration(size_t i, bool useTemporary) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + aLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    const std::vector<double> dataRef = useTemporary ? temporaryData : data;
    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = dataRef[index + j];
    
    return retVec;
}

dvec BubbleManager::getPrevAcceleration(size_t i) const
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + aPrevLoc;
    assert(index < data.size() && "Given index is out of bounds.");

    dvec retVec;
    for (size_t j = 0; j < NUM_DIM; ++j)
        retVec[j] = data[index + j];
    
    return retVec;
}

void BubbleManager::updateAcceleration(size_t i, dvec acceleration)
{
    assert(i < getNumBubbles() && "Given index is out of bounds.");
    size_t index = i * dataStride + aLoc;
    size_t prevIndex = i * dataStride + aPrevLoc;
    assert(index < data.size() && "Given index is out of bounds.");
    assert(prevIndex < data.size() && "Given index is out of bounds.");

    for (size_t j = 0; j < NUM_DIM; ++j)
    {
	// Put the new data to the old, then swap old and current
	// so current contains new and old contains current.
	temporaryData[prevIndex + j] = acceleration[j];
	std::swap(temporaryData[prevIndex + j], temporaryData[index + j]);
    }
}

void BubbleManager::addData(const BubbleData &bubbleData)
{
    for (double val : bubbleData.getValues())
	data.push_back(val);
}
