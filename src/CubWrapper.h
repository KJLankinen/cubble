// -*- C++ -*-

#pragma once

#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <memory>

#include "Env.h"
#include "FixedSizeDeviceArray.h"

namespace cubble
{
class CubWrapper
{
  public:
    CubWrapper(std::shared_ptr<Env> e, size_t numBubbles)
    {
        env = e;
        tempStorage = FixedSizeDeviceArray<char>(numBubbles * sizeof(double), 1);
    }
    ~CubWrapper() {}

    template <typename T, typename InputIterT, typename OutputIterT>
    void reduce(cudaError_t (*func)(void *, size_t &, InputIterT, OutputIterT, int, cudaStream_t, bool),
                InputIterT deviceInputData, OutputIterT deviceOutputData, int numValues, cudaStream_t stream = 0, bool debug = false)
    {
        assert(deviceInputData != nullptr);
        assert(deviceOutputData != nullptr);

        size_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.getDataPtr());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);
    }

    template <typename InputIterT, typename OutputIterT>
    void scan(cudaError_t (*func)(void *, size_t &, InputIterT, OutputIterT, int, cudaStream_t, bool),
              InputIterT deviceInputData, OutputIterT deviceOutputData, int numValues, cudaStream_t stream = 0, bool debug = false)
    {
        assert(deviceInputData != nullptr);
        assert(deviceOutputData != nullptr);

        size_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.getDataPtr());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);
    }

    template <typename KeyT, typename ValueT>
    void sortPairs(cudaError_t (*func)(void *, size_t &, const KeyT *, KeyT *, const ValueT *, ValueT *, int, int, int, cudaStream_t, bool),
                   const KeyT *keysIn, KeyT *keysOut, const ValueT *valuesIn, ValueT *valuesOut, int numValues, cudaStream_t stream = 0, bool debug = false)
    {
        assert(keysIn != nullptr);
        assert(keysOut != nullptr);
        assert(valuesIn != nullptr);
        assert(valuesOut != nullptr);

        size_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, keysIn, keysOut, valuesIn, valuesOut, numValues, 0, sizeof(KeyT) * 8, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = FixedSizeDeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.getDataPtr());
        (*func)(tempStoragePtr, tempStorageBytes, keysIn, keysOut, valuesIn, valuesOut, numValues, 0, sizeof(KeyT) * 8, stream, debug);
    }

  private:
    std::shared_ptr<Env> env;
    FixedSizeDeviceArray<char> tempStorage;
};
} // namespace cubble