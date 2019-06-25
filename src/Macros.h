//-*- C++ -*-
#pragma once

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s

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

// Macro for reading and writing parameters from/to .json file
#define CUBBLE_IO_PARAMETER(read, j, param) \
  do                                        \
  {                                         \
    if (read)                               \
    {                                       \
      param = j[#param];                    \
      std::string s(#param);                \
      s += "Expl";                          \
      std::cout << j[s] << ": " << param;   \
      std::cout << std::endl;               \
    }                                       \
    else                                    \
    {                                       \
      j[#param] = param;                    \
    }                                       \
  } while (0)

#define CUDA_CALL(call) \
  cubble::cudaCallAndLog((call), #call, __FILE__, __LINE__)
#define CUDA_ASSERT(call) \
  cubble::cudaCallAndThrow((call), #call, __FILE__, __LINE__)
#define CURAND_CALL(call) \
  cubble::curandCallAndLog((call), #call, __FILE__, __LINE__)
#define KERNEL_LAUNCH(kernel, ...) \
  cubble::cudaLaunch(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)

// Macro for device assert.
#ifndef NDEBUG
#define DEVICE_ASSERT(statement, msg) \
  cubble::logError(statement, #statement, msg)
#else
#define DEVICE_ASSERT(statement, msg)
#endif

// Macro for (arbitrary) class member variable with getter and setter for CPU
#define CUBBLE_PROP(type, var, defVal)  \
private:                                \
  type var = defVal;                    \
                                        \
public:                                 \
  type get##var() const { return var; } \
  void set##var(const type &val) { var = val; }

// Macro for (arbitrary) const class member variable with getter for CPU
#define CUBBLE_CONST_PROP(type, var, defVal) \
private:                                     \
  type var = defVal;                         \
                                             \
public:                                      \
  type get##var() const { return var; }

// Macro for (arbitrary) class member variable with getter and setter, for CPU &
// GPU
#define CUBBLE_HOST_DEVICE_PROP(type, var, defVal) \
private:                                           \
  type var = defVal;                               \
                                                   \
public:                                            \
  __host__ type get##var() const { return var; }   \
  __host__ void set##var(const type &val) { var = val; }
