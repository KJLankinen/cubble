#pragma once

#include "nlohmann/json.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if (USE_PROFILING == 1)
#define NVTX_RANGE_PUSH_A(string) nvtxRangePushA(string)
#define NVTX_RANGE_POP() nvtxRangePop()
#define CUDA_PROFILER_START(start) startProfiling(start)
#define CUDA_PROFILER_STOP(stop, continue) stopProfiling(stop, continue)
#else
#define NVTX_RANGE_PUSH_A(string)
#define NVTX_RANGE_POP()
#define CUDA_PROFILER_START(start)
#define CUDA_PROFILER_STOP(stop, continue)
#endif

#define CUDA_CALL(call)                                                        \
    cubble::cudaCallAndLog((call), #call, __FILE__, __LINE__)
#define CUDA_ASSERT(call)                                                      \
    cubble::cudaCallAndThrow((call), #call, __FILE__, __LINE__)
#define CURAND_CALL(call)                                                      \
    cubble::curandCallAndLog((call), #call, __FILE__, __LINE__)
#define KERNEL_LAUNCH(kernel, ...)                                             \
    cubble::cudaLaunch(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define CUB_LAUNCH(...) cubble::cubLaunch(__FILE__, __LINE__, __VA_ARGS__)

// Macro for device assert.
#ifndef NDEBUG
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
    using json = nlohmann::json;
    try {
        if (pExc)
            std::rethrow_exception(pExc);
    } catch (const json::exception &e) {
        printf("Encountered a json parse error.\nMake sure the .json file is "
               "correct and filenames are correct.\n%s",
               e.what());
    } catch (const std::exception &e) {
        printf("\n----------Unhandled exception----------\n%s"
               "\n---------------------------------------\n",
               e.what());
        throw e;
    }
}

inline void getFormattedCudaErrorString(cudaError_t result, const char *callStr,
                                        const char *file, int line,
                                        std::basic_ostream<char> &outStream) {
    outStream << "Cuda error encountered."
              << "\n\tType: " << cudaGetErrorName(result)
              << "\n\tDescription: " << cudaGetErrorString(result)
              << "\n\tLocation: " << file << ":" << line
              << "\n\tCall: " << callStr << std::endl;
}

inline bool cudaCallAndLog(cudaError_t result, const char *callStr,
                           const char *file, int line) noexcept {
    if (result != cudaSuccess) {
        getFormattedCudaErrorString(result, callStr, file, line, std::cerr);
        return false;
    }

    return true;
}

inline void cudaCallAndThrow(cudaError_t result, const char *callStr,
                             const char *file, int line) {
    if (result != cudaSuccess) {
        std::stringstream ss;
        getFormattedCudaErrorString(result, callStr, file, line, ss);
        throw std::runtime_error(ss.str());
    }
}

inline bool curandCallAndLog(curandStatus_t result, const char *callStr,
                             const char *file, int line) noexcept {
    if (result != CURAND_STATUS_SUCCESS) {
        std::cerr << "Curand error encountered."
                  << "\n\tType: " << result << "\n\tLocation: " << file << ":"
                  << line << "\n\tCall: " << callStr << std::endl;

        return false;
    }

    return true;
}

inline int getCurrentDeviceAttrVal(cudaDeviceAttr attr) {
#ifndef NDEBUG
    int value = 0;
    int device = 0;

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaDeviceGetAttribute(&value, attr, device));

    return value;
#else
    return -1;
#endif
}

inline void assertMemBelowLimit(const char *kernelStr, const char *file,
                                int line, int bytes, bool abort = true) {
#ifndef NDEBUG
    int value = getCurrentDeviceAttrVal(cudaDevAttrMaxSharedMemoryPerBlock);

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
                                      int line, dim3 blockSize,
                                      bool abort = true) {
#ifndef NDEBUG
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
                                     int line, dim3 gridSize,
                                     bool abort = true) {
#ifndef NDEBUG
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

inline void printRelevantInfoOfCurrentDevice() {
    cudaDeviceProp prop;
    int device = 0;

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));

    printf("\n---------------Properties of current device----------------");
    printf("\n\tGeneral\n\t-------");
    printf("\n\tName: %s", prop.name);
    printf("\n\tCompute capability: %d.%d", prop.major, prop.minor);
    printf("\n\n\tMemory\n\t------");
    printf("\n\tGlobal: %lu B", prop.totalGlobalMem);
    printf("\n\tShared per block: %lu B", prop.sharedMemPerBlock);
    printf("\n\tConstant: %lu B", prop.totalConstMem);
    printf("\n\tRegisters per block: %d", prop.regsPerBlock);
    printf("\n\n\tWarp, threads, blocks, grid\n\t---------------------------");
    printf("\n\tWarp size: %d", prop.warpSize);
    printf("\n\tThreads per block: %d", prop.maxThreadsPerBlock);
    printf("\n\tBlock size: (%d, %d, %d)", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\n\tGrid size: (%d, %d, %d)", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n\tMultiprocessor count: %d\n", prop.multiProcessorCount);
    printf("\nIf you want more info, see %s:%d and"
           "\n'Device Management' section of the CUDA Runtime API docs."
           "\n---------------------------------------------------------\n",
           __FILE__, __LINE__);
}
} // namespace cubble
