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

#pragma once

#include "data_definitions.h"

#include "nlohmann/json.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CUDA_CALL(call)                                                        \
    cubble::cudaCallAndLog((call), #call, __FILE__, __LINE__)
#define CUDA_ASSERT(call)                                                      \
    cubble::cudaCallAndThrow((call), #call, __FILE__, __LINE__)
#define CURAND_CALL(call)                                                      \
    cubble::curandCallAndLog((call), #call, __FILE__, __LINE__)
#define KERNEL_LAUNCH(kernel, ...)                                             \
    cubble::cudaLaunch(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define CUB_LAUNCH(...) cubble::cubLaunch(__FILE__, __LINE__, __VA_ARGS__)

#ifdef PROFILE
#define CUBBLE_PROFILE(start)                                                  \
    if (start) {                                                               \
        cudaProfilerStart();                                                   \
    } else {                                                                   \
        bool stopProfiling = 1000 == params.hostData.numStepsInTimeStep;       \
        stopProfiling |= 2 == params.hostData.timesPrinted &&                  \
                         1 == params.hostData.numStepsInTimeStep;              \
        if (stopProfiling) {                                                   \
            cudaProfilerStop();                                                \
        }                                                                      \
    }
#else
#define CUBBLE_PROFILE(start)
#endif

// Macro for device assert.
#ifdef CUBBLE_DEBUG
#define DEVICE_ASSERT(statement, msg)                                          \
    cubble::logError(statement, #statement, msg)
#else
#define DEVICE_ASSERT(statement, msg)
#endif

// Need to be usable from kernels
#define CUBBLE_PI 3.1415926535897932384626433832795028841971693993
#define CUBBLE_I_PI 1.0 / CUBBLE_PI

namespace cubble {
inline void handleException(const std::exception_ptr pExc) {
    try {
        if (pExc) {
            std::rethrow_exception(pExc);
        }
    } catch (const nlohmann::json::exception &e) {
        printf("Encountered a json parse error.\nMake sure the .json file is "
               "correct and filenames are correct.\n%s\n",
               e.what());
    } catch (const std::exception &e) {
        printf("\n----------Unhandled exception----------\n%s"
               "\n---------------------------------------\n",
               e.what());
        throw e;
    }
}

inline void getFormattedCudaErrorString(cudaError_t result, const char *callStr,
                                        const char *file, int32_t line,
                                        std::basic_ostream<char> &outStream) {
    outStream << "Cuda error encountered."
              << "\n\tType: " << cudaGetErrorName(result)
              << "\n\tDescription: " << cudaGetErrorString(result)
              << "\n\tLocation: " << file << ":" << line
              << "\n\tCall: " << callStr << std::endl;
}

inline bool cudaCallAndLog(cudaError_t result, const char *callStr,
                           const char *file, int32_t line) noexcept {
    if (result != cudaSuccess) {
        getFormattedCudaErrorString(result, callStr, file, line, std::cerr);
        return false;
    }

    return true;
}

inline void cudaCallAndThrow(cudaError_t result, const char *callStr,
                             const char *file, int32_t line) {
    if (result != cudaSuccess) {
        std::stringstream ss;
        getFormattedCudaErrorString(result, callStr, file, line, ss);
        throw std::runtime_error(ss.str());
    }
}

inline bool curandCallAndLog(curandStatus_t result, const char *callStr,
                             const char *file, int32_t line) noexcept {
    if (result != CURAND_STATUS_SUCCESS) {
        std::cerr << "Curand error encountered."
                  << "\n\tType: " << result << "\n\tLocation: " << file << ":"
                  << line << "\n\tCall: " << callStr << std::endl;

        return false;
    }

    return true;
}

inline int32_t getCurrentDeviceAttrVal(cudaDeviceAttr attr) {
#ifdef CUBBLE_DEBUG
    int32_t value = 0;
    int32_t device = 0;

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaDeviceGetAttribute(&value, attr, device));

    return value;
#else
    return -1;
#endif
}

inline void assertMemBelowLimit(const char *kernelStr, const char *file,
                                int32_t line, int32_t bytes,
                                bool abort = true) {
#ifdef CUBBLE_DEBUG
    int32_t value = getCurrentDeviceAttrVal(cudaDevAttrMaxSharedMemoryPerBlock);

    if (bytes > value) {
        std::stringstream ss;
        ss << "Requested size of dynamically allocated shared memory exceeds"
           << " the limitation of the current device."
           << "\nError location: '" << kernelStr << "' @" << file << ":" << line
           << "."
           << "\nRequested size: " << bytes << "\nDevice limit: " << value;

        if (abort) {
            ss << "\nThrowing...";
            throw std::runtime_error(ss.str());
        } else
            std::cerr << ss.str() << std::endl;
    }
#endif
}

inline void assertBlockSizeBelowLimit(const char *kernelStr, const char *file,
                                      int32_t line, dim3 blockSize,
                                      bool abort = true) {
#ifdef CUBBLE_DEBUG
    dim3 temp;
    temp.x = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimX);
    temp.y = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimY);
    temp.z = getCurrentDeviceAttrVal(cudaDevAttrMaxBlockDimZ);

    if (temp.x < blockSize.x || temp.y < blockSize.y || temp.z < blockSize.z) {
        std::stringstream ss;
        ss << "Block size exceeds the limitation of the current device"
           << " in at least one dimension."
           << "\nError location: '" << kernelStr << "' @" << file << ":" << line
           << "."
           << "\nBlock size: (" << blockSize.x << ", " << blockSize.y << ", "
           << blockSize.z << ")"
           << "\nDevice limit: (" << temp.x << ", " << temp.y << ", " << temp.z
           << ")";

        if (abort) {
            ss << "\nThrowing...";
            throw std::runtime_error(ss.str());
        } else
            std::cerr << ss.str() << std::endl;
    }
#endif
}

inline void assertGridSizeBelowLimit(const char *kernelStr, const char *file,
                                     int32_t line, dim3 gridSize,
                                     bool abort = true) {
#ifdef CUBBLE_DEBUG
    dim3 temp;
    temp.x = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimX);
    temp.y = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimY);
    temp.z = getCurrentDeviceAttrVal(cudaDevAttrMaxGridDimZ);

    if (temp.x < gridSize.x || temp.y < gridSize.y || temp.z < gridSize.z) {
        std::stringstream ss;
        ss << "Grid size exceeds the limitation of the current device"
           << " in at least one dimension."
           << "\nError location: '" << kernelStr << "' @" << file << ":" << line
           << "."
           << "\nGrid size: (" << gridSize.x << ", " << gridSize.y << ", "
           << gridSize.z << ")"
           << "\nDevice limit: (" << temp.x << ", " << temp.y << ", " << temp.z
           << ")";

        if (abort) {
            ss << "\nThrowing...";
            throw std::runtime_error(ss.str());
        } else
            std::cerr << ss.str() << std::endl;
    }
#endif
}

template <typename... Arguments>
void cudaLaunch(const char *kernelNameStr, const char *file, int32_t line,
                void (*f)(Arguments...), const Params &params,
                uint32_t sharedMemBytes, cudaStream_t stream,
                Arguments... args) {
#ifdef CUBBLE_DEBUG
    assertMemBelowLimit(kernelNameStr, file, line, sharedMemBytes);
    assertBlockSizeBelowLimit(kernelNameStr, file, line, params.threadBlock);
    assertGridSizeBelowLimit(kernelNameStr, file, line, params.blockGrid);
#endif

    f<<<params.blockGrid, params.threadBlock, sharedMemBytes, stream>>>(
        args...);

#ifdef CUBBLE_DEBUG
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaPeekAtLastError());

    bool errorEncountered = false;
    CUDA_ASSERT(cudaMemcpyFromSymbol(static_cast<void *>(&errorEncountered),
                                     dErrorEncountered, sizeof(bool)));

    if (errorEncountered) {
        std::stringstream ss;
        ss << "Error encountered during kernel execution."
           << "\nError location: '" << kernelNameStr << "' @" << file << ":"
           << line << "."
           << "\nSee earlier messages for possible details.";

        throw std::runtime_error(ss.str());
    }
#endif
}

template <typename... Arguments>
void cubLaunch(const char *file, int32_t line,
               cudaError_t (*func)(void *, size_t &, Arguments...),
               void *tempMem, uint64_t maxMem, Arguments... args) {
    uint64_t tempMemReq = 0;
    (*func)(NULL, tempMemReq, args...);
    if (tempMemReq > maxMem) {
        std::stringstream ss;
        ss << "Not enough temporary memory for cub function call @" << file
           << ":" << line << ".\nRequested " << tempMemReq
           << " bytes, maximum is " << maxMem << " bytes.";
        throw std::runtime_error(ss.str());
    }
    (*func)(tempMem, tempMemReq, args...);
}

} // namespace cubble
