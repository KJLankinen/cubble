#include <iostream>
#include <exception>
#include <stdexcept>
#include <cuda.h>

#include "CubbleApp.h"
#include "Util.h"
#include "Macros.h"

int main(int argc, char **argv)
{
	std::exception_ptr pExc = nullptr;

	int exitCode = EXIT_SUCCESS;

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
		std::string str;
#if (NUM_DIM == 3)
		str = "------------------------------------------------------------------------\n";
		str += "The current program simulates the bubbles in 3D.\n";
		str += "If you want a 2D simulation, change the number of dimensions 'NUM_DIM'";
		str += "\nin Makefile and rebuild the program.\n";
		str += "------------------------------------------------------------------------\n";
#else
		str = "------------------------------------------------------------------------\n";
		str += "The current program simulates the bubbles in 2D.\n";
		str += "If you want a 3D simulation, change the number of dimensions 'NUM_DIM'";
		str += "\nin Makefile and rebuild the program.\n";
		str += "------------------------------------------------------------------------\n";
#endif
		std::cout << str << std::endl;
		cubble::CubbleApp app(argv[1], argv[2]);
		app.run();
	}
	catch (const std::exception &e)
	{
		app.saveSnapshotToFile();
		pExc = std::current_exception();
		cubble::handleException(pExc);

		exitCode = EXIT_FAILURE;
	}

	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaDeviceReset());

	return exitCode;
}
