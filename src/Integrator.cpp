#include <iostream>
#include <sstream>
#include <set>

#include "Integrator.h"
#include "Fileio.h"
#include "Macros.h"
#include "Bubble.h"

using namespace cubble;

Integrator::Integrator(const std::string &inF,
		       const std::string &outF,
		       const std::string &saveF)
{
    inputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + inF);
    outputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + saveF);
    
    readWriteParameters(true);

    dvec interval = tfr - lbb;
    assert(interval[0] == interval[1] && "Simulation box must be a square or a cube!");
#if(NUM_DIM == 3)
    assert(interval[0] == interval[2] && "Simulation box must be a square or a cube!");
#endif
    
    fZeroPerMuZero = sigmaZero * avgRad / muZero;
    generator = std::mt19937(rngSeed);
    uniDist = urdd(0, 1);
    normDist = ndd(avgRad, stdDevRad);

    bubbleData.reserve(numBubblesPerSweep * dataStride);
    nearestNeighbors.reserve(numBubblesPerSweep);
}

Integrator::~Integrator()
{
    readWriteParameters(false);
}

void Integrator::run()
{
    std::cout << "Starting setup." << std::endl;
    setupBubbles();
    std::cout << "Setup done." << std::endl;
}

void Integrator::setupBubbles()
{
    size_t n = 0;
    size_t numGenSweeps = 0;
    size_t numTotalGen = 0;
    double phi = 0;

    auto printInfo = [&]() -> void
	{
	    std::cout << n << " bubbles generated over "
	    << numGenSweeps << " sweeps."
	    << "\nNumber of failed generations: " << numTotalGen - n
	    << "\nVolume fraction: " << phi
	    << "\nTarget volume fraction: " << phiTarget
	    << std::endl;	    
	};
    
    while (true)
    {
	if ((phi - epsilon < phiTarget && phi + epsilon > phiTarget)
	    || numGenSweeps >= numMaxSweeps)
	    break;
	
	for (size_t i = 0; i < numBubblesPerSweep; ++i)
	{
	    generateBubble();
	    numTotalGen++;
	}
	
	updateNearestNeighbors();
	removeIntersectingBubbles();
	n = bubbleData.size() / dataStride;
	phi = getBubbleVolume() / getSimulationBoxVolume();

	if (numGenSweeps % (numMaxSweeps / 100) == 0)
	    printInfo();
	
	numGenSweeps++;
    }

    printInfo();

    std::string filename = "data/bubble_data_2d.dat";
    std::vector<Bubble> temp;
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	dvec pos;
	for (size_t j = 0; j < NUM_DIM; ++j)
	    pos.setComponent(bubbleData[i * dataStride + j], j);
	
	Bubble b(pos, bubbleData[(i + 1) * dataStride - 1]);
	temp.push_back(b);
    }

    fileio::writeVectorToFile(filename, temp);
}

void Integrator::generateBubble()
{
    dvec interval = tfr - lbb;
    double x = uniDist(generator) * interval[0] + lbb[0];
    double y = uniDist(generator) * interval[1] + lbb[1];
#if(NUM_DIM == 3)
    double z = uniDist(generator) * interval[2] + lbb[2];
#endif

    double r = normDist(generator);
    while (r < minRad)
	r = normDist(generator);

    maxRadius = maxRadius < r ? r : maxRadius;
    
    bubbleData.push_back(x);
    bubbleData.push_back(y);
#if(NUM_DIM == 3)
    bubbleData.push_back(z);
#endif
    bubbleData.push_back(r);
}

double Integrator::getSimulationBoxVolume()
{
    dvec temp(tfr - lbb);
    double volume = temp[0] * temp[1];
#if(NUM_DIM == 3)
    volume *= temp[2];
#endif
    return volume;
}

size_t Integrator::getCellIndexFromPos(const dvec &pos, size_t numCellsPerDim)
{
    dvec normedPosVec = (pos - lbb) / (tfr - lbb);
    uvec iv = normedPosVec * numCellsPerDim;

    return getCellIndexFromCellIndexVec(iv, numCellsPerDim);
}

uvec Integrator::getCellIndexVecFromCellIndex(size_t cellIndex,
					      size_t numCellsPerDim)
{
#ifndef NDEBUG
    size_t totalNumCells = 1;
    for (size_t i = 0; i < NUM_DIM; ++i)
	totalNumCells *= numCellsPerDim;
    
    assert(cellIndex < totalNumCells && "Given cellIndex is larger \
than the number of cells per dimension.");
#endif

    uvec temp;
    temp.setComponent(cellIndex % numCellsPerDim, 0);
    temp.setComponent((cellIndex % (numCellsPerDim * numCellsPerDim)) / numCellsPerDim, 1);
#if(NUM_DIM == 3)
    temp.setComponent(cellIndex / (numCellsPerDim * numCellsPerDim), 2);
#endif
    
    return temp;
}

size_t Integrator::getCellIndexFromCellIndexVec(ivec cellIndexVec,
						int numCellsPerDim)
{
    // Periodic boundary conditions:
    int x = cellIndexVec[0];
    x = x > 0
	? (x < numCellsPerDim ? x : x % numCellsPerDim)
	: (x + numCellsPerDim) % numCellsPerDim;
    
    int y = cellIndexVec[1];
    y = y > 0
	? (y < numCellsPerDim ? y : y % numCellsPerDim)
	: (y + numCellsPerDim) % numCellsPerDim;

    size_t index = y * numCellsPerDim + x;
    
#if(NUM_DIM == 3)
    int z = cellIndexVec[2];
    z = z > 0
	? (z < numCellsPerDim ? z : z % numCellsPerDim)
	: (z + numCellsPerDim) % numCellsPerDim;

    index += (size_t)(z * numCellsPerDim * numCellsPerDim);
#endif
    
    return index;
}

void Integrator::updateNearestNeighbors()
{
    size_t numCellsPerDim = std::floor(1.0 / (3.0 * maxRadius / (tfr - lbb)[0]));
    double cellSize = (tfr - lbb)[0] / numCellsPerDim;
    double diam = 2.0 * maxRadius;

    // Reset maxRadius. It's updated later in this function.
    maxRadius = -1.0;

    assert(cellSize > diam && "cellSize is smaller than the diameter \
of the largest bubble.");

    nearestNeighbors.clear();

    std::map<size_t, std::vector<size_t>> cellToBubblesMap;
    std::map<size_t, size_t> bubbleToCellMap;
    std::map<size_t, dvec> bubbleToPositionMap;
    
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	dvec position;
	for (size_t j = 0; j < NUM_DIM; ++j)
	    position.setComponent(bubbleData[i * dataStride + j], j);

	size_t cellIndex = getCellIndexFromPos(position, numCellsPerDim);
	cellToBubblesMap[cellIndex].push_back(i);
	bubbleToCellMap[i] = cellIndex;
	bubbleToPositionMap[i] = position;
    }
    
    assert(bubbleData.size() % dataStride == 0 && "bubbleData has incorrect stride.");
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	nearestNeighbors.emplace_back();

	double radius = bubbleData[(i + 1) * dataStride - 1];
	maxRadius = maxRadius < radius ? radius : maxRadius;
	dvec position = bubbleToPositionMap[i];
	
	ivec cellIndexVec = getCellIndexVecFromCellIndex(bubbleToCellMap[i],
							 numCellsPerDim);
	dvec cellLbb = cellIndexVec.asType<double>() * cellSize;
	dvec cellTfr = (cellIndexVec + 1).asType<double>() * cellSize;

	bool intersectNegX = std::abs(cellLbb[0] - position[0]) < diam;
	bool intersectNegY = std::abs(cellLbb[1] - position[1]) < diam;
	bool intersectPosY = std::abs(cellTfr[1] - position[1]) < diam;
#if(NUM_DIM == 3)
	bool intersectNegZ = std::abs(cellLbb[2] - position[2]) < diam;
	bool intersectPosX = std::abs(cellTfr[0] - position[0]) < diam;
#endif
	
	std::vector<size_t> cellsToSearchNeighborsFrom;
        cellsToSearchNeighborsFrom.push_back(bubbleToCellMap[i]);
	
	if (intersectNegX)
	{
	    ivec temp;
#if(NUM_DIM == 3)
	    temp = cellIndexVec + ivec({-1, 0, 0});
#elif(NUM_DIM == 2)
	    temp = cellIndexVec + ivec({-1, 0});
#else
	    std::cout << "Dimensionality is neither 2D nor 3D..." << std::endl;
#endif
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));

	    if (intersectNegY)
	    {
		ivec temp;
#if(NUM_DIM == 3)
		temp = cellIndexVec + ivec({-1, -1, 0});
#elif(NUM_DIM == 2)
		temp = cellIndexVec + ivec({-1, -1});
#else
		std::cout << "Dimensionality is neither 2D nor 3D..." << std::endl;
#endif
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	    }
	    else if (intersectPosY)
	    {
		ivec temp;
#if(NUM_DIM == 3)
		temp = cellIndexVec + ivec({-1, 1, 0});
#elif(NUM_DIM == 2)
		temp = cellIndexVec + ivec({-1, 1});
#else
		std::cout << "Dimensionality is neither 2D nor 3D..." << std::endl;
#endif
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));	
	    }
	}

	if (intersectNegY)
	{
	    ivec temp;
#if(NUM_DIM == 3)
	    temp = cellIndexVec + ivec({0, -1, 0});
#elif(NUM_DIM == 2)
	    temp = cellIndexVec + ivec({0, -1});
#else
	    std::cout << "Dimensionality is neither 2D nor 3D..." << std::endl;
#endif
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	}

#if(NUM_DIM == 3)
	if (intersectNegZ)
	{
	    ivec temp = cellIndexVec + ivec({0, 0, -1});
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));

	    if (intersectNegX)
	    {	
		ivec temp = cellIndexVec + ivec({-1, 0, -1});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		
		if (intersectNegY)
		{
		    ivec temp = cellIndexVec + ivec({-1, -1, -1});
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		}
		else if (intersectPosY)
		{
		    ivec temp = cellIndexVec + ivec({-1, 1, -1});
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		}
	    }
	    else if (intersectPosX)
	    {
		ivec temp = cellIndexVec + ivec({1, 0, -1});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		if (intersectNegY)
		{
		    ivec temp = cellIndexVec + ivec({1, -1, -1});
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		}
		else if (intersectPosY)
		{
		    ivec temp = cellIndexVec + ivec({1, 1, -1});
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
		}
	    }
	    else if (intersectNegY)
	    {
		ivec temp = cellIndexVec + ivec({0, -1, -1});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	    }
	    else if (intersectPosY)
	    {
		ivec temp = cellIndexVec + ivec({0, 1, -1});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	    }
	}
#endif
        
	for (const size_t &ci : cellsToSearchNeighborsFrom)
	{
	    for (const size_t &j : cellToBubblesMap[ci])
	    {
		if (j == i)
		    continue;
	        
		double radii = bubbleData[(j + 1) * dataStride - 1] + radius;
		dvec pos2;
		for (size_t k = 0; k < NUM_DIM; ++k)
		    pos2.setComponent(bubbleData[j * dataStride + k], k);
		
		if ((position-pos2).getSquaredLength() < radii * radii)
		    nearestNeighbors[i].push_back(j);
	    }
	}
    }
}

void Integrator::removeIntersectingBubbles()
{
    std::set<size_t> toBeDeleted;
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	for (const size_t &j : nearestNeighbors[i])
	{
	    if (toBeDeleted.find(i) == toBeDeleted.end())
		toBeDeleted.insert(j);
	}
    }

    // Remove data from the data vector, from the last item
    // to the first so the indices aren't invalidated.
    for (auto it = toBeDeleted.rbegin(); it != toBeDeleted.rend(); ++it)
    {
	auto b = bubbleData.begin() + *it * dataStride;
	auto e = bubbleData.begin() + (*it + 1) * dataStride;
	
	assert(e <= bubbleData.end() && "Iterator is beyond the end of the vector.");
	
	bubbleData.erase(b, e);
    }
}

double Integrator::getBubbleVolume()
{
    double bubbleVolume = 0;
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	double rad = bubbleData[(i + 1) * dataStride - 1];
	
#if(NUM_DIM == 3)
	rad *= rad * rad;
#elif(NUM_DIM == 2)
	rad *= rad;
#endif
	bubbleVolume += rad;
    }

#if(NUM_DIM == 3)
    bubbleVolume *= 4.0 / 3.0 * M_PI;
#elif(NUM_DIM == 2)
    bubbleVolume *= M_PI;
#else
    std::cout << "Dimensionality of simulation is neither 2D nor 3D..." << std::endl;
#endif

    return bubbleVolume;
}

void Integrator::readWriteParameters(bool read)
{
    std::string msg = read
	? "Reading parameters from file " + inputFile
	: "Saving parameters to file " + saveFile;
    
    std::cout << msg << std::endl;

    nlohmann::json params;

    if (read)
	fileio::readFileToJSON(inputFile, params);

    // When adding new parameters, be sure to add them to the input .json as well
    // and with the exact same name as here.
    
    CUBBLE_PARAMETER(read, params, phiTarget);
    CUBBLE_PARAMETER(read, params, muZero);
    CUBBLE_PARAMETER(read, params, sigmaZero);
    CUBBLE_PARAMETER(read, params, avgRad);
    CUBBLE_PARAMETER(read, params, stdDevRad);
    CUBBLE_PARAMETER(read, params, minRad);
    CUBBLE_PARAMETER(read, params, numBubblesPerSweep);
    CUBBLE_PARAMETER(read, params, lbb);
    CUBBLE_PARAMETER(read, params, tfr);
    CUBBLE_PARAMETER(read, params, errorTolerance);
    CUBBLE_PARAMETER(read, params, timeStep);
    CUBBLE_PARAMETER(read, params, numMaxSweeps);
    CUBBLE_PARAMETER(read, params, rngSeed);
    
    if (!read)
	fileio::writeJSONToFile(saveFile, params);

    std::cout << "Parameter IO done." << std::endl;
}
