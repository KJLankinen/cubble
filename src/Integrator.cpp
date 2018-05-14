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
    inputFile = std::string(_XSTRINGIFY(DATA_PATH) + inF);
    outputFile = std::string(_XSTRINGIFY(DATA_PATH) + outF);
    saveFile = std::string(_XSTRINGIFY(DATA_PATH) + saveF);
    
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
	double rad = bubbleData[i * dataStride + 3];
	rad *= rad * rad;
	bubbleVolume += rad;
    }

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
	Vector3<double> pos(bubbleData[i * dataStride],
			    bubbleData[i * dataStride + 1],
			    bubbleData[i * dataStride + 2]);
	Bubble b(pos, bubbleData[i * dataStride + 3]);
	temp.push_back(b);
    }
    fileio::writeVectorToFile("data/bubble_data.dat", temp);
}

void Integrator::generateBubble()
{
    Vector3<double> interval = tfr - lbb;
    double x = uniDist(generator) * interval.getX() + lbb.getX();
    double y = uniDist(generator) * interval.getY() + lbb.getY();
    double z = uniDist(generator) * interval.getZ() + lbb.getZ();	    

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
    Vector3<double> temp(tfr - lbb);
    return temp.getX() * temp.getY() * temp.getZ();
}

size_t Integrator::getCellIndexFromPos(const Vector3<double> &pos, size_t numCellsPerDim)
{
    Vector3<double> normedPosVec = (pos - lbb) / (tfr - lbb);
    Vector3<size_t> iv = normedPosVec * numCellsPerDim;

    return getCellIndexFromCellIndexVec(iv, numCellsPerDim);
}

Vector3<size_t> Integrator::getCellIndexVecFromCellIndex(size_t cellIndex,
							 size_t numCellsPerDim)
{
    assert(cellIndex < numCellsPerDim * numCellsPerDim * numCellsPerDim);
    return Vector3<size_t>(cellIndex % numCellsPerDim,
			   (cellIndex % (numCellsPerDim * numCellsPerDim)) / numCellsPerDim,
			   cellIndex / (numCellsPerDim * numCellsPerDim));
}

size_t Integrator::getCellIndexFromCellIndexVec(Vector3<size_t> cellIndexVec,
						size_t numCellsPerDim)
{
    // Periodic boundary conditions:
    cellIndexVec %= Vector3<size_t>(1, 1, 1) * numCellsPerDim;
    
    return cellIndexVec.getZ() * numCellsPerDim * numCellsPerDim
	+ cellIndexVec.getY() * numCellsPerDim
	+ cellIndexVec.getX();    
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
    
    // ASSUMPTION: 3 dimensional data
    // ASSMPTION: simulation box is a cube
    size_t numCellsPerDim = std::floor(1.0 / (3.0 * maxRadius / (tfr - lbb).getX()));
    double cellSize = (tfr - lbb).getX() / numCellsPerDim;
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
	Vector3<double> position(bubbleData[i], bubbleData[i + 1], bubbleData[i + 2]);
	size_t cellIndex = getCellIndexFromPos(position, numCellsPerDim);
	cellIndices[cellIndex].push_back(i);

	Vector3<size_t> cellIndexVec = getCellIndexVecFromCellIndex(cellIndex,
								    numCellsPerDim);
	Vector3<double> cellLbb = cellIndexVec * cellSize;
	Vector3<double> cellTfr = (cellIndexVec + 1) * cellSize;

	bool intersectNegX = std::abs(cellLbb.getX() - position.getX()) > diam;
	bool intersectNegY = std::abs(cellLbb.getY() - position.getY()) > diam;
	bool intersectNegZ = std::abs(cellLbb.getZ() - position.getZ()) > diam;
	bool intersectPosX = std::abs(cellTfr.getX() - position.getX()) > diam;
	bool intersectPosY = std::abs(cellTfr.getY() - position.getY()) > diam;
	
	std::vector<size_t> cellsToSearchNeighborsFrom;
        cellsToSearchNeighborsFrom.push_back(cellIndex);
	
	if (intersectNegX)
	{
	    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, 0, 0);
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, numCellsPerDim));

	    // No bubble can simultaneously intersect opposite walls of a cell,
	    // since the cell size is larger than the maximum diameter of all bubbles.
	    if (intersectNegY)
	    {
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, -1, 0);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));
	    }
	    else if (intersectPosY)
	    {
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, 1, 0);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));	
	    }
	}

	if (intersectNegY)
	{
	    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(0, -1, 0);
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, numCellsPerDim));
	}
	
	if (intersectNegZ)
	{
	    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(0, 0, -1);
	    cellsToSearchNeighborsFrom.push_back(
		getCellIndexFromCellIndexVec(temp, numCellsPerDim));

	    if (intersectNegX)
	    {	
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, 0, -1);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		
		if (intersectNegY)
		{
		    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, -1, -1);
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		}
		else if (intersectPosY)
		{
		    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(-1, 1, -1);
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		}
	    }
	    else if (intersectPosX)
	    {
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(1, 0, -1);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		if (intersectNegY)
		{
		    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(1, -1, -1);
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		}
		else if (intersectPosY)
		{
		    Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(1, 1, -1);
		    cellsToSearchNeighborsFrom.push_back(
			getCellIndexFromCellIndexVec(temp, numCellsPerDim));
		}
	    }
	    else if (intersectNegY)
	    {
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(0, -1, -1);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));
	    }
	    else if (intersectPosY)
	    {
		Vector3<size_t> temp = cellIndexVec + Vector3<size_t>(0, 1, -1);
	        cellsToSearchNeighborsFrom.push_back(
		    getCellIndexFromCellIndexVec(temp, numCellsPerDim));
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
	Vector3<double> pos1 = Vector3<double>(bubbleData[i * dataStride],
					       bubbleData[i * dataStride + 1],
					       bubbleData[i * dataStride + 2]);
	double radius = bubbleData[i * dataStride + 3];
	
	for (const size_t &j : tentativeNearestNeighbors[i])
	{
	    Vector3<double> pos2 = Vector3<double>(bubbleData[j * dataStride],
						   bubbleData[j * dataStride + 1],
						   bubbleData[j * dataStride + 2]);
	    double radii = bubbleData[j * dataStride + 3] + radius;

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

    // Remove data from the data vector, from the last item to delete
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
    _PARAMETER(read, params, phiTarget);
    _PARAMETER(read, params, muZero);
    _PARAMETER(read, params, sigmaZero);
    _PARAMETER(read, params, avgRad);
    _PARAMETER(read, params, stdDevRad);
    _PARAMETER(read, params, minRad);
    _PARAMETER(read, params, numBubbles);
    _PARAMETER(read, params, lbb);
    _PARAMETER(read, params, tfr);
    _PARAMETER(read, params, errorTolerance);
    _PARAMETER(read, params, timeStep);

    if (!read)
	fileio::writeJSONToFile(saveFile, params);
}
