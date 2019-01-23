//-*- C++ -*-
#pragma once

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s

#if (USE_PROFILING == 1)
#define NVTX_RANGE_PUSH_A(string) nvtxRangePushA(string)
#define NVTX_RANGE_POP() nvtxRangePop()
#define CUDA_PROFILER_START() cudaProfilerStart()
#define CUDA_PROFILER_STOP() cudaProfilerStop()
#else
#define NVTX_RANGE_PUSH_A(string)
#define NVTX_RANGE_POP()
#define CUDA_PROFILER_START()
#define CUDA_PROFILER_STOP()
#endif

// Macro for reading and writing parameters from/to .json file
#define CUBBLE_IO_PARAMETER(read, j, param)     \
    do                                          \
    {                                           \
        if (read)                               \
        {                                       \
            param = j[#param];                  \
            std::string s(#param);              \
            s += "Expl";                        \
            std::cout << j[s] << ": " << param; \
            std::cout << std::endl;             \
        }                                       \
        else                                    \
        {                                       \
            j[#param] = param;                  \
        }                                       \
    } while (0)

// Cuda error checking
// See Util.h for actual implementation
#ifndef NDEBUG
#define CUDA_CALL(call)                                        \
    do                                                         \
    {                                                          \
        cudaError_t result = call;                             \
        cubble::cudaAssert(result, #call, __FILE__, __LINE__); \
    } while (0)
#else
#define CUDA_CALL(call) \
    do                  \
    {                   \
        call;           \
    } while (0)
#endif

// Curand error checking
// See Util.h for actual implementation
#ifndef NDEBUG
#define CURAND_CALL(result)                                 \
    do                                                      \
    {                                                       \
        cubble::curandAssert((result), __FILE__, __LINE__); \
    } while (0)
#else
#define CURAND_CALL(result) \
    do                      \
    {                       \
        (result);           \
    } while (0)
#endif

#define CUDA_LAUNCH(kernel, ...)                                      \
    do                                                                \
    {                                                                 \
        cudaLaunch(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__); \
    } while (0)

// Macro for device assert.
#ifndef NDEBUG
#define DEVICE_ASSERT(statement, msg)         \
    do                                        \
    {                                         \
        logError(statement, #statement, msg); \
    } while (0)
#else
#define DEVICE_ASSERT(statement, msg)
#endif

// Macro for (arbitrary) class member variable with getter and setter for CPU
#define CUBBLE_PROP(type, var, defVal)    \
  private:                                \
    type var = defVal;                    \
                                          \
  public:                                 \
    type get##var() const { return var; } \
    void set##var(const type &val) { var = val; }

// Macro for (arbitrary) const class member variable with getter for CPU
#define CUBBLE_CONST_PROP(type, var, defVal) \
  private:                                   \
    type var = defVal;                       \
                                             \
  public:                                    \
    type get##var() const { return var; }

// Macro for (arbitrary) class member variable with getter and setter, for CPU & GPU
#define CUBBLE_HOST_DEVICE_PROP(type, var, defVal) \
  private:                                         \
    type var = defVal;                             \
                                                   \
  public:                                          \
    __host__ type get##var() const { return var; } \
    __host__ void set##var(const type &val) { var = val; }
