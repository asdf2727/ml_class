#pragma once

#include <array>
#include <assert.h>

#include "../device.cuh"

namespace device {
	template <typename T>
	struct array;
};

template <typename T>
struct device::array {
	constexpr size_t N;

private:
	T *mem;

public:
	explicit array(const size_t N) : N(N) {
		cudaTry(cudaMalloc(&mem, sizeof(T) * N));
	}
	device::array <T> &operator=(device::array <T> &&other) noexcept {
		assert(N == other.N);
		~array();
		mem = other.mem;
		other.mem = nullptr;
		return *this;
	}

	~array() {
		cudaTry(cudaFree(mem));
	}

	void toHost(T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemcpyAsync(arr, mem, sizeof(T) * N, cudaMemcpyDeviceToHost, stream));
	}
	void fromHost(const T *arr, const cudaStream_t stream = cudaStreamLegacy) {
		cudaTry(cudaMemcpyAsync(mem, arr, sizeof(T) * N, cudaMemcpyHostToDevice, stream));
	}

	void copy(const device::array <T> &arr, const cudaStream_t stream = cudaStreamLegacy) {
		assert(N == arr.N);
		cudaTry(cudaMemcpyAsync(mem, arr.mem, sizeof(T) * N, cudaMemcpyDeviceToDevice, stream));
	}

	// ReSharper disable once CppMemberFunctionMayBeConst
	void set(const uint8_t val, const cudaStream_t stream = cudaStreamLegacy) {
		cudaTry(cudaMemsetAsync(mem, val, sizeof(T) * N, stream));
	}

	operator T *() const {
		return mem;
	}

	device::array <T> &operator= (const T *rhs) {
		fromHost(rhs);
		return *this;
	}
	device::array <T> &operator= (const device::array <T> &rhs) {
		copy(rhs);
		return *this;
	}
};
