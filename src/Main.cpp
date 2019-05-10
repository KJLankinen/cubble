#include <iostream>
#include <exception>
#include <stdexcept>
#include <cuda.h>

#include "Simulator.cuh"
#include "Util.h"

int main(int argc, char **argv)
{
	std::exception_ptr pExc = nullptr;

	if (argc != 3)
	{
		std::cerr << "Two arguments are required."
				  << "\nUsage: " << argv[0] << " inputFile saveFile"
				  << "\ninputFile = the name of the (.json) file that contains"
				  << " the necessary inputs."
				  << "\nsaveFile = file that can be used as a input file"
				  << " to continue from an earlier run"
				  << std::endl;

		return EXIT_FAILURE;
	}

	try
	{
		std::cout << "------------------------------------------------------------------------\n"
				  << "The current program simulates the bubbles in" << NUM_DIM " dimensions.\n"
				  << "If you want to change the dimensionality of the program, change the number of dimensions 'NUM_DIM'"
				  << "\nin Makefile and rebuild the program.\n"
				  << "------------------------------------------------------------------------\n"
				  << std::endl;

		cubble::Simulator simulator;
		simulator.init(argv[1], argv[2]);
		simulator.run();
		simulator.deinit();
	}
	catch (const std::exception &e)
	{
		pExc = std::current_exception();
		cubble::handleException(pExc);

		return EXIT_FAILURE;
	}

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
