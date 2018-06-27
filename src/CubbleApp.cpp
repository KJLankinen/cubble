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
	std::cout << "Starting the shrinking of the simulation box..."
		  << std::flush;
	
	while (phi < phiTarget)
	{
	    // Shrink the simulation box
	    env->setLbb(env->getLbb() + env->getCompressionAmount());
	    env->setTfr(env->getTfr() - env->getCompressionAmount());
	    
	    // Integrate
	    simulator->integrate();
	    
	    // Calculate new phi
	    phi = bubbleVolume / env->getSimulationBoxVolume();
	}
	
	std::cout << " Done." << std::endl;
	printPhi(phi, phiTarget);
	saveSnapshotToFile();
    }
    

    // Stabilize
    std::cout << "\nStarting the stabilization of the foam..." << std::flush;
    // do the stabilization


    // Simulate
    std::cout << " Done\n**Setup done.**"
	      << "\n\n**Starting the simulation proper.**"
	      << std::endl;
    
    while (true)
    {
	// Maybe set a flag in simulator to signify the start of proper simulation?
	simulator->integrate();

	// Dummy break here, until proper end condition
	break;
    }
    
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
       << "\n# bubble data: normalized position (x, y, z),"
       << " unnormalized radius, color as (r, g, b)"
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
