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

#ifdef __NVCC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#define ASSERT_SIZE(struct, bytes)                                             \
    static_assert(sizeof(struct) == bytes,                                     \
                  "size of " #struct " must be equal to " #bytes)

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
