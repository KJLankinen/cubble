// -*- C++ -*-

#pragma once

#include "Macros.h"

#include <memory>
#include <assert.h>
#include <utility>
#include <type_traits>

namespace cubble
{
template <typename T>
class DeviceArray
{
  public:
	DeviceArray(size_t w, size_t h, size_t d)
		: width(w), height(h), depth(d), dataPtr(createDataPtr(w, h, d), destroyDataPtr)
	{
		static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported for now.");
		CUDA_CALL(cudaMemset(static_cast<void *>(dataPtr.get()), 0, sizeof(T) * getSize()));
	}

	DeviceArray(size_t w, size_t h)
		: DeviceArray(w, h, 1)
	{
	}

	DeviceArray()
		: DeviceArray(1, 1, 1)
	{
	}

	~DeviceArray() {}

	T *get() const { return dataPtr.get(); }

	T *getRowPtr(size_t row, size_t slice = 0) const
	{
		assert(row < height);
		assert(slice < depth);

		return dataPtr.get() + slice * width * height + width * row;
	}

	T *getSlicePtr(size_t slice) const
	{
		assert(slice < depth);

		return dataPtr.get() + slice * width * height;
	}

	size_t getWidth() const { return width; }
	size_t getHeight() const { return height; }
	size_t getDepth() const { return depth; }
	size_t getSliceSize() const { return width * height; }
	size_t getSize() const { return width * height * depth; }
	size_t getSizeInBytes() const { return sizeof(T) * getSize(); }

	// Strictly speaking this should only be allowed when T is an integer type...
	void setBytesToZero() { CUDA_CALL(cudaMemset(static_cast<void *>(dataPtr.get()), 0, getSizeInBytes())); }

	void operator=(DeviceArray<T> &&o)
	{
		width = o.width;
		height = o.height;
		depth = o.depth;
		dataPtr = std::move(o.dataPtr);
	}

  private:
	T *createDataPtr(size_t w, size_t h, size_t d)
	{
		T *t;
		CUDA_CALL(cudaMalloc((void **)&t, w * h * d * sizeof(T)));

		return t;
	}

	static void destroyDataPtr(T *t)
	{
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaFree(static_cast<void *>(t));
	}

	size_t width = 0;
	size_t height = 0;
	size_t depth = 0;

	std::unique_ptr<T[], decltype(&destroyDataPtr)> dataPtr;
};
}; // namespace cubble
