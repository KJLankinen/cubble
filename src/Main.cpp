#include <iostream>
#include <exception>
#include <stdexcept>

#include "Simulator.h"
#include "Util.h"

int main(int argc, char **argv)
{
    std::exception_ptr pExc = nullptr;

    if (argc != 4)
    {
	std::cerr << "Three arguments are required."
		  << "\nUsage: " << argv[0] << " inputFile outputFile saveFile"
		  << "\ninputFile = the name of the (.json) file that contains"
		  << " the necessary inputs."
		  << "\noutputFile = file where the program output is written to"
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
	cubble::Simulator simulator(argv[1], argv[2], argv[3]);
        simulator.run();
    }
    catch (const std::exception &e)
    {
	pExc = std::current_exception();
	cubble::handleException(pExc);
	
	return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
