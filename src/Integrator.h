#pragma once

#include <string>

namespace cubble
{
    // This class is the workhorse of the simulation.
    class Integrator
    {
    public:
	Integrator(const std::string &inputFile,
		   const std::string &outputFile,
		   const std::string &saveFile);
	
	~Integrator();

	void integrate(double dt);
	
    private:
	//----------
	// Functions
	//----------
	void serialize();
	void deserialize();

	// ----------
	// Parameters
	//-----------

	// filenames
	std::string inputFile;
	std::string outputFile;
	std::string saveFile;

	// physical parameters
	double phi = 0.0;
	double alpha = 0.0;
    };
}
