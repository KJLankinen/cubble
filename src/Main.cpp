#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cubble {
void run(std::string &&inputFileName);
}

int main(int argc, char **argv) {
    std::exception_ptr pExc = nullptr;

    if (argc != 2) {
        std::cout << "\nUsage: " << argv[0] << " inputFile"
                  << "\ninputFile = the name of the (.json) file that contains"
                  << " the necessary inputs" << std::endl;

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
