#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cubble
{
void run(std::string &&inputFileName, std::string &&outputFileName);
}

int main(int argc, char **argv)
{
  std::exception_ptr pExc = nullptr;

  if (argc != 3)
  {
    std::cout
      << "\nUsage: " << argv[0] << " inputFile outputFile"
      << "\ninputFile = the name of the (.json) file that contains"
      << " the necessary inputs, or the name of the binary file that contains the serialized state of a non-finished "
         "simulation.\noutputFile = (.bin) file name where to save data if simulation ends before completion"
      << std::endl;

    return EXIT_FAILURE;
  }

  try
  {
    std::cout << "-------------------------------------------------------------"
                 "-----------\n"
              << "The current program simulates the bubbles in " << NUM_DIM << " dimensions.\n"
              << "If you want to change the dimensionality of the program, "
                 "change the number of dimensions 'NUM_DIM'"
              << "\nin Makefile and rebuild the program.\n"
              << "-------------------------------------------------------------"
                 "-----------\n"
              << std::endl;

    cubble::run(std::string(argv[1]), std::string(argv[2]));
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
