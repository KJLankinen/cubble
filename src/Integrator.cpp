#include <iostream>

#include "Integrator.h"
#include "Fileio.h"
#include "Macros.h"

using namespace cubble;

class Integrator::Cell
{
public:
    Cell() {}
    ~Cell() {}

    void addNeighbor(Cell *cell)
    {
	if (cell)
	    neighbors.push_back(cell);
	else
	    std::cerr << "Cell to add was a nullptr!" << std::endl;
    }

    void addBubble(Bubble *bubble)
    {
	if (bubble)
	    bubbles.push_back(bubble);
	else
	    std::cerr << "Bubble to add was a nullptr!" << std::endl;
    }

    void getOwnBubbles(std::vector<Bubble*> &b) const
    {
        b = bubbles;
    }

    void getAllBubbles(std::vector<Bubble*> &b) const
    {
	b.clear();
	b.insert(b.end(), bubbles.begin(), bubbles.end());
	
	std::vector<Bubble*> temp;
	for (const Cell *c : neighbors)
	{
	    c->getOwnBubbles(temp);
	    b.insert(b.end(), temp.begin(), temp.end());
	}
    }
    
private:
    std::vector<Cell*> neighbors;
    std::vector<Bubble*> bubbles;
};

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
    bubbles.reserve(numBubbles);
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
	generateBubble(Vector3<double>(0, 0, 0), Vector3<double>(1, 1, 1));

    fileio::writeVectorToFile(outputFile, bubbles);
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
	    c.addNeighbor(&cells[getIndexFromCoords(xm, i, z)]);
	    
	    for (size_t j : xVec)
		c.addNeighbor(&cells[getIndexFromCoords(j, i, zm)]);
	}

	c.addNeighbor(&cells[getIndexFromCoords(x, ym, z)]);
    }
}

void Integrator::generateBubble(Vector3<double> intervalStart, Vector3<double> intervalEnd)
{
    auto generatePosition = [=]() -> Vector3<double>
	{
	    Vector3<double> position(uniDist(generator),
				     uniDist(generator),
				     uniDist(generator));
	    
	    position *= intervalEnd - intervalStart;
	    position += intervalStart;

	    return position;
	};

    Vector3<double> position = generatePosition();
    double radius = normDist(generator);
    while (radius < minRad)
	radius = normDist(generator);

    Bubble bubble(position, radius);
    bubbles.push_back(bubble);
}

void Integrator::integrate(double dt)
{
    phi = dt * 0.1;
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

    // These parameters can be found from the .json that's given as the inputFile
    _PARAMETERIZE(read, phi, params);
    _PARAMETERIZE(read, avgRad, params);
    _PARAMETERIZE(read, stdDevRad, params);
    _PARAMETERIZE(read, minRad, params);
    _PARAMETERIZE(read, numBubbles, params);
    _PARAMETERIZE(read, cellsPerDim, params);

    if (!read)
	fileio::writeJSONToFile(saveFile, params);
}
