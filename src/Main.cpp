#include <iostream>
#include <exception>
#include <stdexcept>
#include <cuda.h>

#include "Simulator.cuh"
#include "Util.h"

int main(int argc, char **argv)
{
	std::exception_ptr pExc = nullptr;

	if (argc != 2)
	{
		std::cerr << "One argument are required."
				  << "\nUsage: " << argv[0] << " inputFile"
				  << "\ninputFile = the name of the (.json) file that contains"
				  << " the necessary inputs."
				  << std::endl;

		return EXIT_FAILURE;
	}

	try
	{
		std::cout << "------------------------------------------------------------------------\n"
				  << "The current program simulates the bubbles in "
				  << NUM_DIM
				  << " dimensions.\n"
				  << "If you want to change the dimensionality of the program, change the number of dimensions 'NUM_DIM'"
				  << "\nin Makefile and rebuild the program.\n"
				  << "------------------------------------------------------------------------\n"
				  << std::endl;

		cubble::Simulator simulator;
		simulator.init(argv[1]);
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
