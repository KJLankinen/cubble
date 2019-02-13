#pragma once

#include "Macros.h"

#include <memory>
#include <assert.h>
#include <utility>
#include <type_traits>
#include <cuda_runtime_api.h>

namespace cubble
{

template <typename T>
class PinnedHostArray
{
  public:
    PinnedHostArray(size_t size)
        : size(size), dataPtr(createDataPtr(size), destroyDataPtr)
    {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported for now.");
        // Strictly speaking, this doesn't guarantee that anything else than integers are zero...
        memset(static_cast<void *>(dataPtr.get()), 0, sizeof(T) * size);
    }

    PinnedHostArray()
        : PinnedHostArray(1)
    {
    }

    ~PinnedHostArray() {}

    T *get() const { return dataPtr.get(); }
    size_t getSize() const { return size; }
    size_t getSizeInBytes() const { return sizeof(T) * getSize(); }

    void operator=(PinnedHostArray<T> &&o)
    {
        size = o.size;
        dataPtr = std::move(o.dataPtr);
    }

  private:
    T *createDataPtr(size_t size)
    {
        T *t = nullptr;
        CUDA_ASSERT(cudaMallocHost((void **)&t, size * sizeof(T)));

        return t;
    }

    static void destroyDataPtr(T *t)
    {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFreeHost(static_cast<void *>(t)));
    }

    size_t size = 0;

    std::unique_ptr<T[], decltype(&destroyDataPtr)> dataPtr;
};
}; // namespace cubble