// -*- C++ -*-

#pragma once

#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <memory>

#include "Env.h"
#include "DeviceArray.h"

namespace cubble
{
class CubWrapper
{
  public:
    CubWrapper(std::shared_ptr<Env> e, size_t numBubbles)
    {
        env = e;

        outData = DeviceArray<char>(sizeof(double), 1);
        tempStorage = DeviceArray<char>(numBubbles * sizeof(double), 1);
    }
    ~CubWrapper() {}

    template <typename T, typename InputIterT, typename OutputIterT>
    T reduce(cudaError_t (*func)(void *, size_t &, InputIterT, OutputIterT, int, cudaStream_t, bool),
             InputIterT deviceInputData, int numValues, cudaStream_t stream = 0, bool debug = false)
    {
        assert(deviceInputData != nullptr);

        if (sizeof(T) > outData.getSizeInBytes())
            outData = DeviceArray<char>(sizeof(T), 1);

        void *rawOutputPtr = static_cast<void *>(outData.get());
        OutputIterT deviceOutputData = static_cast<OutputIterT>(rawOutputPtr);

        size_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);

        T hostOutputData;
        cudaMemcpyAsync(&hostOutputData, deviceOutputData, sizeof(T), cudaMemcpyDeviceToHost, stream);

        return hostOutputData;
    }

    template <typename T, typename InputIterT, typename OutputIterT>
    void reduceNoCopy(cudaError_t (*func)(void *, size_t &, InputIterT, OutputIterT, int, cudaStream_t, bool),
                      InputIterT deviceInputData, OutputIterT deviceOutputData, int numValues, cudaStream_t stream = 0, bool debug = false)
    {
        assert(deviceInputData != nullptr);
        assert(deviceOutputData != nullptr);

        size_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData, numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
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
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
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
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, keysIn, keysOut, valuesIn, valuesOut, numValues, 0, sizeof(KeyT) * 8, stream, debug);
    }

  private:
    std::shared_ptr<Env> env;
    DeviceArray<char> outData;
    DeviceArray<char> tempStorage;
};
} // namespace cubble