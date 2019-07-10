#pragma once

#include "Util.h"
#include <assert.h>
#include <memory>

namespace cubble
{
template <typename T>
class DeviceArray
{
public:
  DeviceArray(uint64_t w = 0, uint64_t h = 1, uint64_t d = 1)
    : width(w)
    , height(h)
    , depth(d)
    , dataPtr(createDataPtr(w, h, d), destroyDataPtr)
  {
  }

  DeviceArray(const DeviceArray<T> &other)
    : DeviceArray(other.width, other.height, other.depth)
  {
    CUDA_ASSERT(cudaMemcpy(static_cast<void *>(dataPtr.get()), static_cast<void *>(other.dataPtr.get()),
                           other.getSizeInBytes(), cudaMemcpyDeviceToDevice));
  }

  DeviceArray(DeviceArray<T> &&other)
    : DeviceArray()
  {
    swap(*this, other);
  }

  ~DeviceArray() {}

  DeviceArray<T> &operator=(DeviceArray<T> other)
  {
    swap(*this, other);

    return *this;
  }

  friend void swap(DeviceArray<T> &first, DeviceArray<T> &second)
  {
    using std::swap;

    swap(first.width, second.width);
    swap(first.height, second.height);
    swap(first.depth, second.depth);
    swap(first.dataPtr, second.dataPtr);
  }

  T *get() const { return dataPtr.get(); }

  T *getRowPtr(uint64_t row, uint64_t slice = 0) const
  {
    assert(row < height);
    assert(slice < depth);

    return dataPtr.get() + slice * width * height + width * row;
  }

  T *getSlicePtr(uint64_t slice) const
  {
    assert(slice < depth);

    return dataPtr.get() + slice * width * height;
  }

  uint64_t getWidth() const { return width; }
  uint64_t getHeight() const { return height; }
  uint64_t getDepth() const { return depth; }
  uint64_t getSliceSize() const { return width * height; }
  uint64_t getSize() const { return width * height * depth; }
  uint64_t getSizeInBytes() const { return sizeof(T) * getSize(); }
  void setBytesToZero() { CUDA_ASSERT(cudaMemset(static_cast<void *>(dataPtr.get()), 0, getSizeInBytes())); }

private:
  static T *createDataPtr(uint64_t w, uint64_t h, uint64_t d)
  {
    T *t = nullptr;
    if (w * h * d > 0)
      CUDA_ASSERT(cudaMalloc((void **)&t, w * h * d * sizeof(T)));

    return t;
  }

  static void destroyDataPtr(T *t)
  {
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaFree(static_cast<void *>(t)));
  }

  uint64_t width  = 0;
  uint64_t height = 0;
  uint64_t depth  = 0;

  std::unique_ptr<T[], decltype(&destroyDataPtr)> dataPtr;
};
}; // namespace cubble
