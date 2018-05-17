#include <iostream>
#include <sstream>
#include <set>

#include "Simulator.h"
#include "Fileio.h"
#include "Macros.h"

using namespace cubble;

Simulator::Simulator(const std::string &inF,
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
    nearestNeighbors.reserve(numBubblesPerSweep);
    bubbleManager = std::make_shared<BubbleManager>(numMaxBubbles,
						    rngSeed,
						    avgRad,
						    stdDevRad,
						    minRad);
}

Simulator::~Simulator()
{
    readWriteParameters(false);
}

void Simulator::run()
{
    std::cout << "Starting setup." << std::endl;
    setupSimulation();
    std::cout << "Setup done." << std::endl;
}

void Simulator::setupSimulation()
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

    bool targetVolumeFractionReached = false;
    while (true)
    {
	if (phi >= phiTarget)
	{
	    std::cout << "Target volume fraction reached."
		      << " Stopping bubble generation."
		      << std::endl;
	    
	    targetVolumeFractionReached = true;
	    
	    break;
	}
	else if (numGenSweeps >= numMaxSweeps)
	{
	    std::cout << "Maximum number of generation sweeps reached."
		      << " Stopping bubble generation."
		      << std::endl;
	    break;   
	}
	else if (n >= numMaxBubbles)
	{
	    std::cout << "Maximum number of bubbles reached."
		      << " Stopping bubble generation."
		      << std::endl;
	    break;   
	}
	
	for (size_t i = 0; i < numBubblesPerSweep; ++i)
	{
	    double newRad = bubbleManager->generateBubble();
	    maxRadius = newRad > maxRadius ? newRad : maxRadius;
	    numTotalGen++;
	}
	
	updateNearestNeighbors();
	removeIntersectingBubbles();
	phi = getBubbleVolume() / getSimulationBoxVolume();
        n = bubbleManager->getNumBubbles();
	
	numGenSweeps++;
    }

    bubbleManager->updateTemporary();
    
    printInfo();

    if (!targetVolumeFractionReached)
    {
	std::cout << "Couldn't reach the target volume fraction by generating bubbles."
		  << "\nStarting compression of simulation box."
		  << std::endl;

	while (!targetVolumeFractionReached)
	{
	    phi = getBubbleVolume() / getSimulationBoxVolume();
	    targetVolumeFractionReached = phi >= phiTarget;

	    compressSimulationBox();
	}

	std::cout << "Compression done."
		  << "\nTarget volume fraction: " << phiTarget
		  << "\nVolume fraction: " << phi
		  << std::endl;
    }

    std::string filename = "data/bubble_data_2d.dat";
    std::vector<vec<double, NUM_DIM + 1>> temp;
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
    {
	dvec pos = getScaledPosition(bubbleManager->getPosition(i));
	vec<double, NUM_DIM + 1> bubble;
	for (size_t j = 0; j < NUM_DIM; ++j)
	    bubble[j] = pos[j];
	
        bubble[NUM_DIM] = bubbleManager->getRadius(i);
	temp.push_back(bubble);
    }

    fileio::writeVectorToFile(filename, temp);
}

void Simulator::integrate()
{
    auto predict = [](dvec a, dvec b, dvec c, double dt) -> dvec
	{
	    return a + 0.5 * dt * (3.0 * b - c);
	};

    auto correct = [](dvec a, dvec b, dvec c, double dt) -> dvec
	{
	    return a + 0.5 * dt * (b + c);
	};

    double error = 0.0;
    
    do
    {
	computeAcceleration();
	
	for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	{
	    bubbleManager->updatePosition(i, wrapAroundBoundaries(
					      predict(bubbleManager->getPosition(i),
						      bubbleManager->getVelocity(i),
						      bubbleManager->getPrevVelocity(i),
						      timeStep)));
	    
	    bubbleManager->updateVelocity(i, predict(bubbleManager->getVelocity(i),
						     bubbleManager->getAcceleration(i),
						     bubbleManager->getPrevAcceleration(i),
						     timeStep));
	}

	computeAcceleration();

	for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	{
	    bubbleManager->updatePosition(i, wrapAroundBoundaries(
					      correct(bubbleManager->getPosition(i),
						      bubbleManager->getVelocity(i, true),
						      bubbleManager->getVelocity(i),
						      timeStep)));
	    
	    bubbleManager->updateVelocity(i, correct(bubbleManager->getVelocity(i),
						  bubbleManager->getAcceleration(i, true),
						  bubbleManager->getAcceleration(i),
						  timeStep));
	}

	if (error < errorTolerance / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > errorTolerance)
	    timeStep *= 0.5;
    }
    while(error > errorTolerance);

    bubbleManager->swapData();
}

void Simulator::computeAcceleration()
{
    
}

dvec Simulator::wrapAroundBoundaries(dvec position)
{
    for (size_t i = 0; i < NUM_DIM; ++i)
    {
	double val = position[i];
	double interval = tfr[i] - lbb[i];
        position[i] = val >= lbb[i] ? (val <= tfr[i] ? val : val - interval) : val + interval;
    }

    return position;
}

double Simulator::getSimulationBoxVolume()
{
    dvec temp(tfr - lbb);
    double volume = temp[0] * temp[1];
#if(NUM_DIM == 3)
    volume *= temp[2];
#endif
    return volume;
}

size_t Simulator::getCellIndexFromNormalizedPosition(const dvec &pos,
						     size_t numCellsPerDim)
{
    uvec iv = pos * numCellsPerDim;

    return getCellIndexFromCellIndexVec(iv, numCellsPerDim);
}

uvec Simulator::getCellIndexVecFromCellIndex(size_t cellIndex,
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
    temp[0] = cellIndex % numCellsPerDim;
    temp[1] = (cellIndex % (numCellsPerDim * numCellsPerDim)) / numCellsPerDim;
#if(NUM_DIM == 3)
    temp[2] = cellIndex / (numCellsPerDim * numCellsPerDim);
#endif
    
    return temp;
}

size_t Simulator::getCellIndexFromCellIndexVec(ivec cellIndexVec,
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

void Simulator::updateNearestNeighbors()
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
    
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
    {
	size_t cellIndex = getCellIndexFromNormalizedPosition(bubbleManager->getPosition(i),
							      numCellsPerDim);
	
	cellToBubblesMap[cellIndex].push_back(i);
	bubbleToCellMap[i] = cellIndex;
    }
    
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
    {
	nearestNeighbors.emplace_back();

	dvec position = getScaledPosition(bubbleManager->getPosition(i));
	double radius = bubbleManager->getRadius(i);
	maxRadius = maxRadius < radius ? radius : maxRadius;
	
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
	        
		double radii = bubbleManager->getRadius(j) + radius;
		dvec pos2 = getScaledPosition(bubbleManager->getPosition(j));
		
		if ((position-pos2).getSquaredLength() < radii * radii)
		    nearestNeighbors[i].push_back(j);
	    }
	}
    }
}

void Simulator::removeIntersectingBubbles()
{
    std::set<size_t> toBeDeleted;
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
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
	bubbleManager->removeData(*it);
}

double Simulator::getBubbleVolume()
{
    double bubbleVolume = 0;
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
    {
	double rad = bubbleManager->getRadius(i);
	
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

dvec Simulator::getScaledPosition(const dvec &position)
{
    return position * (tfr - lbb) + lbb;
}

void Simulator::compressSimulationBox()
{
    double halfCompr = 0.5 * compressionAmount;
    lbb += halfCompr;
    tfr -= halfCompr;
}

void Simulator::readWriteParameters(bool read)
{
    std::string msg = read
	? "Reading parameters from file " + inputFile
	: "Saving parameters to file " + saveFile;
    
    std::cout << msg << "\n" << std::endl;

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
    CUBBLE_PARAMETER(read, params, numMaxBubbles);
    CUBBLE_PARAMETER(read, params, compressionAmount);
    
    if (!read)
	fileio::writeJSONToFile(saveFile, params);

    std::cout << "\nParameter IO done." << std::endl;
}
