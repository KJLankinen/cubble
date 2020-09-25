#include "Util.h"
#include <cuda.h>
#include <exception>
#include <stdexcept>
#include <stdio.h>
#include <string>

namespace cubble {
void run(std::string &&inputFileName);
}

int main(int argc, char **argv) {
    std::exception_ptr pExc = nullptr;

    if (argc != 2) {
        printf("Usage: %s inputFile,\nwhere inputFile is the name of the "
               "(.json) file containing the simulation input.\n",
               argv[0]);

        return EXIT_FAILURE;
    }

    int numGPUs = 0;
    CUDA_CALL(cudaGetDeviceCount(&numGPUs));
    if (1 > numGPUs) {
        printf("No CUDA capable devices found.\n");
        return EXIT_FAILURE;
    }

    try {
        cubble::run(std::string(argv[1]));
    } catch (const std::exception &e) {
        pExc = std::current_exception();
        cubble::handleException(pExc);

        return EXIT_FAILURE;
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
