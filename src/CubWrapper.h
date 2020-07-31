#pragma once

#include "DeviceArray.h"
#include <cuda_runtime.h>
#include <memory>
#include <nvToolsExt.h>

namespace cubble {
class CubWrapper {
  public:
    CubWrapper() {
        outData = DeviceArray<char>(sizeof(double));
        tempStorage = DeviceArray<char>(1024);
    }
    ~CubWrapper() {}

    template <typename T, typename InputIterT, typename OutputIterT>
    T reduce(cudaError_t (*func)(void *, uint64_t &, InputIterT, OutputIterT,
                                 int, cudaStream_t, bool),
             InputIterT deviceInputData, int numValues, cudaStream_t stream = 0,
             bool debug = false) {
        assert(deviceInputData != nullptr);

        if (sizeof(T) > outData.getSizeInBytes())
            outData = DeviceArray<char>(sizeof(T), 1);

        void *rawOutputPtr = static_cast<void *>(outData.get());
        OutputIterT deviceOutputData = static_cast<OutputIterT>(rawOutputPtr);

        uint64_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData,
                numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData,
                deviceOutputData, numValues, stream, debug);

        T hostOutputData;
        cudaMemcpyAsync(&hostOutputData, deviceOutputData, sizeof(T),
                        cudaMemcpyDeviceToHost, stream);

#ifndef NDEBUG
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaPeekAtLastError());
#endif

        return hostOutputData;
    }

    template <typename T, typename InputIterT, typename OutputIterT>
    void reduceNoCopy(cudaError_t (*func)(void *, uint64_t &, InputIterT,
                                          OutputIterT, int, cudaStream_t, bool),
                      InputIterT deviceInputData, OutputIterT deviceOutputData,
                      int numValues, cudaStream_t stream = 0,
                      bool debug = false) {
        assert(deviceInputData != nullptr);
        assert(deviceOutputData != nullptr);

        uint64_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData,
                numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData,
                deviceOutputData, numValues, stream, debug);

#ifndef NDEBUG
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaPeekAtLastError());
#endif
    }

    template <typename InputIterT, typename OutputIterT>
    void scan(cudaError_t (*func)(void *, uint64_t &, InputIterT, OutputIterT,
                                  int, cudaStream_t, bool),
              InputIterT deviceInputData, OutputIterT deviceOutputData,
              int numValues, cudaStream_t stream = 0, bool debug = false) {
        assert(deviceInputData != nullptr);
        assert(deviceOutputData != nullptr);

        uint64_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, deviceInputData, deviceOutputData,
                numValues, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, deviceInputData,
                deviceOutputData, numValues, stream, debug);

#ifndef NDEBUG
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaPeekAtLastError());
#endif
    }

    template <typename KeyT, typename ValueT>
    void sortPairs(cudaError_t (*func)(void *, uint64_t &, const KeyT *, KeyT *,
                                       const ValueT *, ValueT *, int, int, int,
                                       cudaStream_t, bool),
                   const KeyT *keysIn, KeyT *keysOut, const ValueT *valuesIn,
                   ValueT *valuesOut, int numValues, cudaStream_t stream = 0,
                   bool debug = false) {
        assert(keysIn != nullptr);
        assert(keysOut != nullptr);
        assert(valuesIn != nullptr);
        assert(valuesOut != nullptr);

        uint64_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, keysIn, keysOut, valuesIn, valuesOut,
                numValues, 0, sizeof(KeyT) * 8, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, keysIn, keysOut, valuesIn,
                valuesOut, numValues, 0, sizeof(KeyT) * 8, stream, debug);

#ifndef NDEBUG
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaPeekAtLastError());
#endif
    }

    template <typename SampleIteratorT, typename CounterT, typename LevelT,
              typename OffsetT>
    void histogram(cudaError_t (*func)(void *, uint64_t &, SampleIteratorT,
                                       CounterT *, int, LevelT, LevelT, OffsetT,
                                       cudaStream_t, bool),
                   SampleIteratorT samples, CounterT *deviceOutHist,
                   int numLevels, LevelT lowerLevel, LevelT upperLevel,
                   OffsetT numSamples, cudaStream_t stream = 0,
                   bool debug = false) {
        assert(deviceOutHist != nullptr);
        assert(samples != nullptr);

        uint64_t tempStorageBytes = 0;
        (*func)(NULL, tempStorageBytes, samples, deviceOutHist, numLevels,
                lowerLevel, upperLevel, numSamples, stream, debug);

        if (tempStorageBytes > tempStorage.getSizeInBytes())
            tempStorage = DeviceArray<char>(tempStorageBytes, 1);

        void *tempStoragePtr = static_cast<void *>(tempStorage.get());
        (*func)(tempStoragePtr, tempStorageBytes, samples, deviceOutHist,
                numLevels, lowerLevel, upperLevel, numSamples, stream, debug);

#ifndef NDEBUG
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaPeekAtLastError());
#endif
    }

  private:
    DeviceArray<char> outData;
    DeviceArray<char> tempStorage;
};
} // namespace cubble
