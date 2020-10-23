/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
