#pragma once

#include "Macros.h"

#include <memory>
#include <assert.h>

namespace cubble
{

template <typename T>
class PinnedHostArray
{
  public:
    PinnedHostArray(size_t size = 0)
        : size(size), dataPtr(createDataPtr(size), destroyDataPtr)
    {
    }

    PinnedHostArray(const PinnedHostArray<T> &other)
        : PinnedHostArray(other.size)
    {
        memcpy(static_cast<void *>(dataPtr.get()), static_cast<void *>(other.dataPtr.get()), other.getSizeInBytes());
    }

    PinnedHostArray(PinnedHostArray<T> &&other)
        : PinnedHostArray()
    {
        swap(*this, other);
    }

    ~PinnedHostArray() {}

    PinnedHostArray<T> &operator=(PinnedHostArray<T> other)
    {
        swap(*this, other);

        return *this;
    }

    friend void swap(PinnedHostArray<T> &first, PinnedHostArray<T> &second)
    {
        using std::swap;

        swap(first.size, second.size);
        swap(first.dataPtr, second.dataPtr);
    }

    T *get() const { return dataPtr.get(); }
    size_t getSize() const { return size; }
    size_t getSizeInBytes() const { return sizeof(T) * getSize(); }

  private:
    static T *createDataPtr(size_t size)
    {
        T *t = nullptr;
        if (size > 0)
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