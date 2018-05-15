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
		       const std::string &saveF,
		       const int rngSeed)
{
    inputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + inF);
    outputFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(CUBBLE_XSTRINGIFY(DATA_PATH) + saveF);
    
    readWriteParameters(true);

    fZeroPerMuZero = sigmaZero * avgRad / muZero;
    generator = std::mt19937(rngSeed);
    uniDist = urdd(0, 1);
    normDist = ndd(avgRad, stdDevRad);

    bubbleData.reserve(numBubbles * dataStride);
    nearestNeighbors.reserve(numBubbles);
    tentativeNearestNeighbors.reserve(numBubbles);
}

Integrator::~Integrator()
{
    readWriteParameters(false);
}

void Integrator::run()
{
    size_t n = 0;
    size_t numGenSweeps = 0;
    size_t numTotalGen = 0;
    while (true)
    {
	if (n == numBubbles || numGenSweeps > 100)
	    break;
	
	for (size_t i = n; i < numBubbles; ++i)
	{
	    generateBubble();
	    numTotalGen++;
	}

	updateNearestNeighbors();
	removeIntersectingBubbles();
	n = bubbleData.size() / dataStride;
	
	numGenSweeps++;
    }

    double bubbleVolume = 0;
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	// ASSUMPTION: 3-dimensional data
	double rad = bubbleData[(i + 1) * dataStride - 1];
	rad *= rad * rad;
	bubbleVolume += rad;
    }

    // ASSUMPTION: 3-dimensional data
    bubbleVolume *= 4.0 / 3.0 * M_PI;

    std::cout << "Generated " << n
	      << " bubbles over " << numGenSweeps
	      << " sweeps."
	      << "\nNumber of failed generations: " << numTotalGen - n
	      << "\nVolume fraction: " << bubbleVolume / getSimulationBoxVolume()
	      << std::endl;

    std::vector<Bubble> temp;
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	// ASSUMPTION: 3-dimensional data
	dvec pos = {bubbleData[i * dataStride],
		    bubbleData[i * dataStride + 1],
		    bubbleData[i * dataStride + 2]};
	Bubble b(pos, bubbleData[(i + 1) * dataStride - 1]);
	temp.push_back(b);
    }
    
    fileio::writeVectorToFile("data/bubble_data.dat", temp);
}

void Integrator::generateBubble()
{
    // ASSUMPTION: 3-dimensional data
    dvec interval = tfr - lbb;
    double x = uniDist(generator) * interval[0] + lbb[0];
    double y = uniDist(generator) * interval[1] + lbb[1];
    double z = uniDist(generator) * interval[2] + lbb[2];	    

    double r = normDist(generator);
    while (r < minRad)
	r = normDist(generator);

    maxRadius = maxRadius < r ? r : maxRadius;
    
    bubbleData.push_back(x);
    bubbleData.push_back(y);
    bubbleData.push_back(z);
    bubbleData.push_back(r);
}

double Integrator::getSimulationBoxVolume()
{
    // ASSUMPTION: 3-dimensional data
    dvec temp(tfr - lbb);
    return temp[0] * temp[1] * temp[2];
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
    // ASSUMPTION: 3-dimensional data
    assert(cellIndex < numCellsPerDim * numCellsPerDim * numCellsPerDim);
    return uvec({cellIndex % numCellsPerDim,
		(cellIndex % (numCellsPerDim * numCellsPerDim)) / numCellsPerDim,
		cellIndex / (numCellsPerDim * numCellsPerDim)});
}

size_t Integrator::getCellIndexFromCellIndexVec(ivec cellIndexVec,
						int numCellsPerDim)
{
    // ASSUMPTION: 3-dimensional data
    
    // Periodic boundary conditions:
    int x = cellIndexVec[0];
    x = x > 0
	? (x < numCellsPerDim ? x : x % numCellsPerDim)
	: (x + numCellsPerDim) % numCellsPerDim;
    
    int y = cellIndexVec[1];
    y = y > 0
	? (y < numCellsPerDim ? y : y % numCellsPerDim)
	: (y + numCellsPerDim) % numCellsPerDim;
    
    int z = cellIndexVec[2];
    z = z > 0
	? (z < numCellsPerDim ? z : z % numCellsPerDim)
	: (z + numCellsPerDim) % numCellsPerDim;
    
    return (size_t)(z * numCellsPerDim * numCellsPerDim + y * numCellsPerDim + x);
}

void Integrator::updateNearestNeighbors()
{
    /*
     * This function finds all the neighbors for each bubble that are
     * closer than the maximum diameter.
     *
     * The search is done in steps, each step refining and decreasing the domain
     * from which to search for the neighbors.
     *
     * First, the entire simulation box is divided into cells, the size of which
     * is slightly larger than the maximum diameter of all the bubbles.
     * Then each cell wall is checked for intersection: if a bubble is closer
     * to the cell wall than the maximum diameter, the intersected neighboring cell
     * is added to the cells that are loop over when looking for the intersecting bubbles.
     *
     * Note that this is done s.t. each cell only looks for neighbors in all the 9 cells
     * directly below, the 3 cells directly to the left and the one cell directly behind.
     * This way every possible bubble interaction is only taken into account once:
     * the bubbles above this bubble's cell handle the intersection with this bubble
     * when they're the current bubble.
     *
     * Finally, only the bubbles that intersect this bubble are added to the nearerst
     * neighbors vector for later use when the forces and velocities are calculated.
     */
    
    // ASSUMPTION: 3-dimensional data
    // ASSMPTION: simulation box is a cube
    size_t numCellsPerDim = std::floor(1.0 / (3.0 * maxRadius / (tfr - lbb)[0]));
    double cellSize = (tfr - lbb)[0] / numCellsPerDim;
    double diam = 2.0 * maxRadius;

    // If everything works correctly, this should always be true.
    assert(cellSize > diam);

    nearestNeighbors.clear();
    tentativeNearestNeighbors.clear();

    std::vector<std::vector<size_t>> cellIndices;
    cellIndices.reserve(numCellsPerDim * numCellsPerDim * numCellsPerDim);

    for (size_t i = 0; i < numCellsPerDim * numCellsPerDim * numCellsPerDim; ++i)
	cellIndices.emplace_back();
    
    assert(bubbleData.size() % dataStride == 0);
    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	tentativeNearestNeighbors.emplace_back();
	dvec position = {bubbleData[i], bubbleData[i + 1], bubbleData[i + 2]};
	size_t cellIndex = getCellIndexFromPos(position, numCellsPerDim);
	cellIndices[cellIndex].push_back(i);

	ivec cellIndexVec = getCellIndexVecFromCellIndex(cellIndex,
								    numCellsPerDim);
	dvec cellLbb = cellIndexVec * cellSize;
	dvec cellTfr = (cellIndexVec + 1) * cellSize;

	bool intersectNegX = std::abs(cellLbb[0] - position[0]) > diam;
	bool intersectNegY = std::abs(cellLbb[1] - position[1]) > diam;
	bool intersectNegZ = std::abs(cellLbb[2] - position[2]) > diam;
	bool intersectPosX = std::abs(cellTfr[0] - position[0]) > diam;
	bool intersectPosY = std::abs(cellTfr[1] - position[1]) > diam;
	
	std::vector<size_t> cellsToSearchNeighborsFrom;
        cellsToSearchNeighborsFrom.push_back(cellIndex);
	
	if (intersectNegX)
	{
	    ivec temp = cellIndexVec + ivec({-1, 0, 0});
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));

	    // No bubble can simultaneously intersect opposite walls of a cell,
	    // since the cell size is larger than the maximum diameter of all bubbles.
	    if (intersectNegY)
	    {
		ivec temp = cellIndexVec + ivec({-1, -1, 0});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	    }
	    else if (intersectPosY)
	    {
		ivec temp = cellIndexVec + ivec({-1, 1, 0});
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));	
	    }
	}

	if (intersectNegY)
	{
	    ivec temp = cellIndexVec + ivec({0, -1, 0});
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, (int)numCellsPerDim));
	}
	
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

	// Temporarily add the cell indices, not the bubble indices.
	nearestNeighbors.push_back(cellsToSearchNeighborsFrom);
    }

    // Add all the bubbles from all the cells that were closer
    // than the maximum diameter.
    for (size_t i = 0; i < nearestNeighbors.size(); ++i)
    {
	const auto &cells = nearestNeighbors[i];
	for (const size_t &cellIndex : cells)
	{
	    for (const size_t &bubbleIndex : cellIndices[cellIndex])
	    {
		if (bubbleIndex == i)
		    continue;
		
		tentativeNearestNeighbors[i].push_back(bubbleIndex);
	    }
	}

	// Remove the temporary cell indices, they aren't needed anymore.
	nearestNeighbors[i].clear();
    }

    for (size_t i = 0; i < bubbleData.size() / dataStride; ++i)
    {
	dvec pos1 = {bubbleData[i * dataStride],
		     bubbleData[i * dataStride + 1],
		     bubbleData[i * dataStride + 2]};
	double radius = bubbleData[(i + 1) * dataStride - 1];
	
	for (const size_t &j : tentativeNearestNeighbors[i])
	{
	    dvec pos2 = {bubbleData[j * dataStride],
			 bubbleData[j * dataStride + 1],
			 bubbleData[j * dataStride + 2]};
	    double radii = bubbleData[(j + 1) * dataStride - 1] + radius;

	    if ((pos1-pos2).getSquaredLength() < radii * radii)
		nearestNeighbors[i].push_back(j);
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
	    if (toBeDeleted.find(i) == toBeDeleted.end() &&
		toBeDeleted.find(j) == toBeDeleted.end())
		toBeDeleted.insert(j);
	}
    }

    // Remove data from the data vector, from the last item
    // to the first so the indices aren't invalidated.
    for (auto it = toBeDeleted.end(); it != toBeDeleted.begin(); --it)
    {
	auto b = bubbleData.begin() + *it;
	auto e = bubbleData.begin() + *it + dataStride;
	
	assert(e < bubbleData.end());
	
	bubbleData.erase(b, e);
    }
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
    CUBBLE_PARAMETER(read, params, numBubbles);
    CUBBLE_PARAMETER(read, params, lbb);
    CUBBLE_PARAMETER(read, params, tfr);
    CUBBLE_PARAMETER(read, params, errorTolerance);
    CUBBLE_PARAMETER(read, params, timeStep);
    
    if (!read)
	fileio::writeJSONToFile(saveFile, params);
}
