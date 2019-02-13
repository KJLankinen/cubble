#pragma once

#include "Macros.h"

#include <memory>
#include <assert.h>

namespace cubble
{
template <typename T>
class DeviceArray
{
  public:
	DeviceArray(size_t w = 0, size_t h = 0, size_t d = 0)
		: width(w), height(h), depth(d), dataPtr(createDataPtr(w, h, d), destroyDataPtr)
	{
	}

	DeviceArray(size_t w, size_t h)
		: DeviceArray(w, h, 1)
	{
	}

	DeviceArray(const DeviceArray<T> &other)
		: DeviceArray(other.width, other.height, other.depth)
	{
		CUDA_ASSERT(cudaMemcpy(static_cast<void *>(dataPtr.get()),
							   static_cast<void *>(other.dataPtr.get()),
							   other.getSizeInBytes(),
							   cudaMemcpyDeviceToDevice));
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
	void setBytesToZero() { CUDA_ASSERT(cudaMemset(static_cast<void *>(dataPtr.get()), 0, getSizeInBytes())); }

  private:
	static T *createDataPtr(size_t w, size_t h, size_t d)
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

	size_t width = 0;
	size_t height = 0;
	size_t depth = 0;

	std::unique_ptr<T[], decltype(&destroyDataPtr)> dataPtr;
};
}; // namespace cubble
