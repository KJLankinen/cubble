#pragma once

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

#define CUDA_CALL(call) cubble::cudaCallAndLog((call), #call, __FILE__, __LINE__)
#define CUDA_ASSERT(call) cubble::cudaCallAndThrow((call), #call, __FILE__, __LINE__)
#define CURAND_CALL(call) cubble::curandCallAndLog((call), #call, __FILE__, __LINE__)
#define KERNEL_LAUNCH(kernel, ...) cubble::cudaLaunch(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)

// Macro for device assert.
#ifndef NDEBUG
#define DEVICE_ASSERT(statement, msg) cubble::logError(statement, #statement, msg)
#else
#define DEVICE_ASSERT(statement, msg)
#endif
