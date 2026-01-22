#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace nanochat {

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// Simple GPU tensor wrapper
template <typename T>
class Tensor {
public:
    T* data = nullptr;
    size_t size = 0;

    Tensor() = default;

    explicit Tensor(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(T)));
    }

    ~Tensor() {
        if (data) cudaFree(data);
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void allocate(size_t n) {
        if (data) cudaFree(data);
        size = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(T)));
    }

    void from_host(const T* host_data, size_t n) {
        if (size != n) allocate(n);
        CUDA_CHECK(cudaMemcpy(data, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void to_host(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() {
        CUDA_CHECK(cudaMemset(data, 0, size * sizeof(T)));
    }
};

}  // namespace nanochat
