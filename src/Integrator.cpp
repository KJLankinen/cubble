#include <iostream>
#include <memory>

#include "Integrator.h"
#include "Fileio.h"
#include "Macros.h"

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

    generator = std::mt19937(rngSeed);
    uniDist = urdd(0, 1);
    normDist = ndd(avgRad, stdDevRad);
    prepareCells();
}

Integrator::~Integrator()
{
    readWriteParameters(false);
}

void Integrator::run()
{
    // Start the simulation.
    // Do the necessary pre-simulation steps and then start
    // the 'main loop'.

    for (size_t i = 0; i < numBubbles; ++i)
	generateBubble();

    std::vector<Bubble> bubblesVec;
    for (const auto &pair : bubbles)
	bubblesVec.push_back(pair.second);
    fileio::writeVectorToFile(outputFile, bubblesVec);
    
    std::vector<Bubble*> bubbleRefs;
    for (const auto &c : cells)
	c.getBubbleRefsAsVector(bubbleRefs);
    
    fileio::writeVectorToFile("data/bubble_refs.dat", bubbleRefs);

    removeBubble(bubbles[0]);
}

void Integrator::prepareCells()
{
    auto getIndexFromCoords = [=](size_t x, size_t y, size_t z) -> size_t
	{
	    return z * cellsPerDim * cellsPerDim + y * cellsPerDim + x;
	};
    
    for (size_t i = 0; i < cellsPerDim * cellsPerDim * cellsPerDim; ++i)
	cells.emplace_back();

    // Add neighbors to each cell.
    // This needs to be done only once,
    // since the cells themselves don't move or change,
    // only their contents do.
    // The coordinates are s.t. bottom back left is the first cell,
    // and travesing from left to right, back to front, bottom to top.
    for (size_t n = 0; n < cells.size(); ++n)
    {
	// Calculate 3D coordinates from index
	size_t x = n % cellsPerDim;
	size_t y = n / cellsPerDim;
	size_t z = n / (cellsPerDim * cellsPerDim);
	
	size_t xp = x < cellsPerDim - 1 ? x + 1 : 0;
	size_t yp = y < cellsPerDim - 1 ? y + 1 : 0;
	size_t xm = x > 0 ? x - 1 : cellsPerDim - 1;
	size_t ym = y > 0 ? y - 1 : cellsPerDim - 1;
	size_t zm = z > 0 ? z - 1 : cellsPerDim - 1;

	std::vector<size_t> xVec = {xm, x, xp};
	std::vector<size_t> yVec = {ym, y, yp};

	Cell &c = cells[n];

	// Add overall 13 neigbors:
	// The 3 x 3 below current cell (z - 1),
	// the left side at same z-coordinate
	// and one on same z- and x-coordinates but y - 1.

	// The order of the neighboring cells in the neighbor vector of
	// the cells doesn't really matter, since either all or none
	// of the neighbors are used.
	for (size_t i : yVec)
	{
	    c.addNeighborRef(&cells[getIndexFromCoords(xm, i, z)]);
	    
	    for (size_t j : xVec)
		c.addNeighborRef(&cells[getIndexFromCoords(j, i, zm)]);
	}

	c.addNeighborRef(&cells[getIndexFromCoords(x, ym, z)]);
    }
}

void Integrator::generateBubble()
{
    auto generatePosition = [=](Vector3<size_t> &cellIndex) -> Vector3<double>
	{
	    Vector3<double> position(uniDist(generator),
				     uniDist(generator),
				     uniDist(generator));

	    cellIndex = position * cellsPerDim;
	    position *= tfr - lbb;
	    position += lbb;

	    return position;
	};

    Vector3<size_t> cellIndex;
    Vector3<double> position = generatePosition(cellIndex);
    
    double radius = normDist(generator);
    while (radius < minRad)
	radius = normDist(generator);

    size_t i = cellIndex.getZ() * cellsPerDim * cellsPerDim
	+ cellIndex.getY() * cellsPerDim
	+ cellIndex.getX();

    Bubble bubble(position, radius);
    bubble.setCellIndex(i);
    bubbles[bubble.getUID()] = bubble;
    
    cells[i].addBubbleRef(&bubbles[bubble.getUID()]);
}

void Integrator::removeBubble(const Bubble &bubble)
{
    cells[bubble.getCellIndex()].removeBubbleRef(bubble.getUID());
    bubbles.erase(bubble.getUID());
}

void Integrator::integrate(double dt)
{
    phiTarget = dt * 0.1;
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
    _PARAMETER(read, params, avgRad);
    _PARAMETER(read, params, stdDevRad);
    _PARAMETER(read, params, minRad);
    _PARAMETER(read, params, numBubbles);
    _PARAMETER(read, params, cellsPerDim);
    _PARAMETER(read, params, lbb);
    _PARAMETER(read, params, tfr);

    if (!read)
	fileio::writeJSONToFile(saveFile, params);
}
