#include <iostream>

#include "CubbleApp.h"
#include "Fileio.h"
#include "Bubble.h"

using namespace cubble;

int CubbleApp::numSnapshots = 0;

CubbleApp::CubbleApp(const std::string &inF,
		     const std::string &saveF)
{
    env = std::make_shared<Env>(inF, saveF);
    env->readParameters();
    
    simulator = std::make_unique<Simulator>(env);
}

CubbleApp::~CubbleApp()
{
    env->writeParameters();
}

void CubbleApp::run()
{
    std::cout << "**Starting the simulation setup.**\n" << std::endl;
    simulator->setupSimulation();

    // Bubble volume doesn't change while the simulation box is being shrunk.
    const double bubbleVolume = simulator->getVolumeOfBubbles();
    const double phiTarget = env->getPhiTarget();
    double phi = bubbleVolume / env->getSimulationBoxVolume();

    auto printPhi = [](double phi, double phiTarget) -> void
	{
	    std::cout << "Volume ratios: current: " << phi
	    << ", target: " << phiTarget
	    << std::endl;
	};

    printPhi(phi, phiTarget);
    saveSnapshotToFile();
    
    // Shrink the box and integrate bubbles until at target phi
    if (phi < phiTarget)
    {
        int shrinkCount = 0;
	std::cout << "Starting the shrinking of the simulation box." << std::endl;
	
	while (phi < phiTarget)
	{
	    // Shrink the simulation box
	    env->setLbb(env->getLbb() + env->getCompressionAmount());
	    env->setTfr(env->getTfr() - env->getCompressionAmount());
	    
	    // Integrate
	    simulator->integrate();
	    
	    // Calculate new phi
	    phi = bubbleVolume / env->getSimulationBoxVolume();
	    
	    ++shrinkCount;

	    if (shrinkCount % 100 == 0)
		simulator->assignBubblesToCells();
	}
	
	std::cout << "Shrinkin took total of " << shrinkCount << " steps." << std::endl;
	printPhi(phi, phiTarget);
	saveSnapshotToFile();
    }
    
    // Stabilize
    int numRelaxationSteps = 0;
    std::cout << "\nStarting the stabilization of the foam..." << std::flush;
    while (true)
    {
	double energy1 = simulator->getElasticEnergy();
	double time = 0;

	for (int i = 0; i < env->getNumStepsToRelax(); ++i)
	{
	    simulator->integrate();
	    time += env->getTimeStep();
	    
	    if (i % 100 == 0)
		simulator->assignBubblesToCells();
	}
	
	double energy2 = simulator->getElasticEnergy();
	double deltaEnergy = std::abs(energy2 - energy1) / (energy1 * time);

	if (deltaEnergy < env->getMaxDeltaEnergy())
	{
	    std::cout << "Final delta energy " << deltaEnergy
		      << " after " << (numRelaxationSteps + 1) * env->getNumStepsToRelax()
		      << " steps."
		      << std::endl;
	    break;
	}
	else
	    std::cout << "Number of simulation steps relaxed: "
		      << numRelaxationSteps * env->getNumStepsToRelax()
		      << ", delta energy: " << deltaEnergy
		      << std::endl;

	++numRelaxationSteps;
    }
    saveSnapshotToFile();

    // Simulate
    std::cout << " Done\n**Setup done.**"
	      << "\n\n**Starting the simulation proper.**"
	      << std::endl;
    
    for (int i = 0; i < env->getNumIntegrationSteps(); ++i)
    {
	// Maybe set a flag in simulator to signify the start of proper simulation?
	simulator->integrate();
	
	if (i % 100 == 0)
	    simulator->assignBubblesToCells();
    }
    
    simulator->assignBubblesToCells();

    saveSnapshotToFile();
    
    std::cout << "**Simulation has been finished.**\nGoodbye!" << std::endl;
}

void CubbleApp::saveSnapshotToFile()
{
    std::cout << "Writing a snap shot to a file..." << std::flush;

    std::vector<Bubble> tempVec;
    simulator->getBubbles(tempVec);
    
    std::stringstream ss;
    ss << env->getDataPath()
       << env->getSnapshotFilename()
       << numSnapshots
       << ".dat";

    std::string filename(ss.str());
    ss.clear();
    ss.str("");

    // Add descriptions here, when adding new things to the 'header' of the data file
    ss << "#--------------------------------------------------"
       << "\n# Lines starting with '#' are comment lines"
       << "\n#"
       << "\n# Format of data:"
       << "\n# left bottom back"
       << "\n# top front right"
       << "\n#"
       << "\n# bubble data: normalized position (x, y, z), unnormalized radius"
       << "\n#--------------------------------------------------";

    // Add the new things here.
    ss << "\n" << env->getLbb()
       << "\n" << env->getTfr();
    
    for (const auto &bubble : tempVec)
	ss << "\n" << bubble;
    
    fileio::writeStringToFile(filename, ss.str());
    ++numSnapshots;

    std::cout << " Done." << std::endl;
}
