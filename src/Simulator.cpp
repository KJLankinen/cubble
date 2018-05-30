#include <iostream>
#include <sstream>
#include <set>

#include "Simulator.h"
#include "Fileio.h"
#include "Macros.h"
#include "CudaContainer.h"
#include "CudaKernelsWrapper.h"

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
    accelerations.reserve(numMaxBubbles);
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
    CudaContainer<float, 3> testContainer;
    CudaKernelsWrapper cudaKernelsWrapper;
    cudaKernelsWrapper.testFunctionWrapper();
    /*
    std::cout << "Starting setup." << std::endl;
    setupSimulation();
    std::cout << "Setup done." << std::endl;
    
    std::string filename = "data/bubble_data_2d_setup.dat";
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

    std::cout << "Starting integration." << std::endl;
    for (size_t i = 0; i < numIntegrationSteps; ++i)
	integrate();
    std::cout << "Integration finished." << std::endl;

    filename = "data/bubble_data_2d_integrated.dat";
    temp.clear();
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
    filename = "data/acc.dat";
    fileio::writeVectorToFile(filename, accelerations);
    */
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
	    maxDiameter = 2.0 * newRad > maxDiameter ? 2.0 * newRad : maxDiameter;
	    numTotalGen++;
	}
        
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
	    integrate();
	}

	std::cout << "Compression done."
		  << "\nTarget volume fraction: " << phiTarget
		  << "\nVolume fraction: " << phi
		  << std::endl;
    }
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

    double error = -1.0;
    accelerations.resize(bubbleManager->getNumBubbles());
    do
    {
	resetCellData();
	maxDiameter = -1.0;
	error = -1.0;
	
	// Predict new position based on current velocities
	for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	{
	    dvec position = getNormalizedPosition(
		wrapAroundBoundaries(
		    predict(getScaledPosition(bubbleManager->getPosition(i)),
			    bubbleManager->getVelocity(i),
			    bubbleManager->getPrevVelocity(i),
			    timeStep)));
	    
	    bubbleManager->updatePosition(i, position);
	    updateCellDataForBubble(i, position);
	    
	    // Update maximum diameter
	    maxDiameter = maxDiameter < 2.0 * bubbleManager->getRadius(i)
		? 2.0 * bubbleManager->getRadius(i)
		: maxDiameter;
	}

	// Update the accelerations
	dvec zeroVec;
	std::fill(accelerations.begin(), accelerations.end(), zeroVec);
	for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	{
	    dvec position = getScaledPosition(bubbleManager->getPosition(i, true));
	    double radius = bubbleManager->getRadius(i, true);

	    // For every neighboring cell (own cell included), 14 in 3D, 5 in 2D
	    // See 'addNeighboringCellsToVec()' for details.
	    bool ownCell = true;
	    for (size_t ci : bubbleToCells[i])
	    {
		double multiplier = ownCell ? 0.5 : 1.0;
		// For every bubble in a cell, on average numBubbles/numCells
		for (size_t j : cellToBubbles[ci])
		{
		    // Skip self
		    if (j == i)
			continue;
		    
		    dvec position2 = getScaledPosition(bubbleManager->getPosition(j, true));
		    double radii = bubbleManager->getRadius(j, true) + radius;

		    dvec acceleration = getShortestVecBetween(position, position2);
		    // Skip bubbles that don't overlap
		    if (acceleration.getSquaredLength() > radii * radii)
			continue;
		    
		    double magnitude = acceleration.getLength();
		    acceleration *= (radii - magnitude) / (radii * magnitude);
		    accelerations[i] += multiplier * acceleration;
		    accelerations[j] -= multiplier * acceleration;
		}
	    
		ownCell = false;
	    }
	}

	// Take the correction step, using the original position,
	// velocity based on the predicted position and current velocity.
	for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	{
	    // Calculate velocity based on new accelerations.
	    dvec velocity = fZeroPerMuZero * accelerations[i];
	    bubbleManager->updateVelocity(i, velocity);
	
	    dvec position = wrapAroundBoundaries(
		correct(getScaledPosition(bubbleManager->getPosition(i)),
			bubbleManager->getVelocity(i, true),
			bubbleManager->getVelocity(i),
			timeStep));
	    
	    dvec predictedPos = getScaledPosition(bubbleManager->getPosition(i, true));

	    // Compute error
	    for (size_t j = 0; j < NUM_DIM; ++j)
	    {
		double diff = std::abs(predictedPos[j] - position[j]);
		error = diff > error ? diff : error;
	    }
	    
	    bubbleManager->updatePosition(i, getNormalizedPosition(position));
	}

	if (error < errorTolerance / 10 && timeStep < 0.1)
	    timeStep *= 1.9;
	else if (error > errorTolerance)
	    timeStep *= 0.5;
    }
    while(error > errorTolerance);

    bubbleManager->swapData();

    integrationTime += timeStep;
    
    // Handle the rest of the integration related things,
    // e.g. redistributing gas from small bubbles,
    // adding timeStep to simulation time, etc.
}

dvec Simulator::wrapAroundBoundaries(dvec position)
{
    // Using scaled position
    for (size_t i = 0; i < NUM_DIM; ++i)
    {
	double val = position[i];
	double interval = tfr[i] - lbb[i];
        position[i] = val >= lbb[i]
	    ? (val <= tfr[i] ? val : val - interval)
	    : val + interval;
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

size_t Simulator::getCellIndexFromNormalizedPosition(const dvec &pos)
{
    uvec iv = pos * numCellsPerDim;

    return getCellIndexFromCellIndexVec(iv);
}

uvec Simulator::getCellIndexVecFromCellIndex(size_t cellIndex)
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

size_t Simulator::getCellIndexFromCellIndexVec(ivec cellIndexVec)
{
    int ncpd = (int)numCellsPerDim;
    // Periodic boundary conditions:
    int x = cellIndexVec[0];
    x = x > 0
	? (x < ncpd ? x : x % ncpd)
	: (x + ncpd) % ncpd;
    
    int y = cellIndexVec[1];
    y = y > 0
	? (y < ncpd ? y : y % ncpd)
	: (y + ncpd) % ncpd;

    size_t index = y * ncpd + x;
    
#if(NUM_DIM == 3)
    int z = cellIndexVec[2];
    z = z > 0
	? (z < ncpd ? z : z % ncpd)
	: (z + ncpd) % ncpd;

    index += (size_t)(z * numCellsPerDim * numCellsPerDim);
#endif
    
    return index;
}

void Simulator::removeIntersectingBubbles()
{
    // This function is only used at the stat of the simulation.
    
    resetCellData();

    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
	updateCellDataForBubble(i, bubbleManager->getPosition(i));
    
    std::set<size_t> toBeDeleted;
    for (size_t i = 0; i < bubbleManager->getNumBubbles(); ++i)
    {
	dvec position = getScaledPosition(bubbleManager->getPosition(i));
	double radius = bubbleManager->getRadius(i);
	
	for (size_t ci : bubbleToCells[i])
	{
	    for (size_t j : cellToBubbles[ci])
	    {
		// Skip self
		if (j == i)
		    continue;
		
		dvec position2 = getScaledPosition(bubbleManager->getPosition(j));
		double radii = bubbleManager->getRadius(j) + radius;
		
		dvec diff = getShortestVecBetween(position, position2);
		if (diff.getSquaredLength() > radii * radii)
		    continue;
		
		if (toBeDeleted.find(i) == toBeDeleted.end())
		    toBeDeleted.insert(j);
	    }
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

dvec Simulator::getNormalizedPosition(const dvec &position)
{
    return (position - lbb) / (tfr - lbb);
}

void Simulator::compressSimulationBox()
{
    double halfCompr = 0.5 * compressionAmount;
    lbb += halfCompr;
    tfr -= halfCompr;
}

void Simulator::addNeighborCellsToVec(std::vector<size_t> &v, size_t cellIndex)
{
    ivec cellIndexVec = getCellIndexVecFromCellIndex(cellIndex);
#if(NUM_DIM == 3)
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  0,  0})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1, -1,  0})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  1,  0})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 0, -1,  0})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 0,  0, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  0, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1, -1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 1,  0, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 1, -1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 1,  1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 0, -1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 0,  1, -1})));
#elif(NUM_DIM == 2)
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  0})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1, -1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({-1,  1})));
    v.push_back(getCellIndexFromCellIndexVec(cellIndexVec + ivec({ 0, -1})));
#else
    std::cout << "Dimensionality is neither 2D nor 3D..." << std::endl;
#endif
}

void Simulator::resetCellData()
{
    numCellsPerDim = std::floor(1.0 / (1.5 * maxDiameter / (tfr - lbb)[0]));
    size_t numTotalCells = 1;
    for (size_t i = 0; i < NUM_DIM; ++i)
	numTotalCells *= numCellsPerDim;
    
    cellToBubbles.clear();
    cellToBubbles.resize(numTotalCells);
    
    bubbleToCells.clear();
    bubbleToCells.resize(bubbleManager->getNumBubbles());   
}

void Simulator::updateCellDataForBubble(size_t i, dvec position)
{
    // Update cell's bubbles
    size_t cellIndex = getCellIndexFromNormalizedPosition(position);
    cellToBubbles[cellIndex].push_back(i);
    
    // Update bubble's neighboring cells
    bubbleToCells[i].push_back(cellIndex);
    addNeighborCellsToVec(bubbleToCells[i], cellIndex);
}

dvec Simulator::getShortestVecBetween(dvec position1, dvec position2)
{
    // This function calculates returns the shortest 'path' (= vector) between
    // two positions while taking into account periodic boundary conditions.

    dvec interval = tfr - lbb;
    dvec temp = position1 - position2;

    for (size_t i = 0; i < NUM_DIM; ++i)
    {
        if (std::abs(temp[i]) > 0.5 * interval[i])
	    position2[i] = temp[i] < 0 ? temp[i] + interval[i] : temp[i] - interval[i];
    }

    return position1 - position2;
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
    CUBBLE_PARAMETER(read, params, numIntegrationSteps);
    
    if (!read)
	fileio::writeJSONToFile(saveFile, params);

    std::cout << "\nParameter IO done." << std::endl;
}
