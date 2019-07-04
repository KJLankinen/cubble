#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace cubble
{
void run(const char *inputFileName, const char *outputFileName);
}

int main(int argc, char **argv)
{
  std::exception_ptr pExc = nullptr;

  if (argc != 2)
  {
    std::cout << "\nUsage: " << argv[0] << " inputFile"
              << "\ninputFile = the name of the (.json) file that contains"
              << " the necessary inputs." << std::endl;

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

    cubble::run(argv[1]);
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
