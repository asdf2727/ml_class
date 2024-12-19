#pragma once

#include <array>
#include <assert.h>

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

	void copy(const device::array <T> &arr, const cudaStream_t stream = cudaStreamLegacy) const {
		assert(N == arr.N);
		cudaTry(cudaMemcpyAsync(mem, arr.mem, sizeof(T) * N, cudaMemcpyDeviceToDevice, stream));
	}
	device::array <T> &operator= (const device::array <T> &rhs) const {
		copy(rhs);
		return *this;
	}

	~array() {
		cudaTry(cudaFree(mem));
	}

	void toHost(T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemcpyAsync(arr, mem, sizeof(T) * N, cudaMemcpyDeviceToHost, stream));
	}
	void fromHost(const T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemcpyAsync(mem, arr, sizeof(T) * N, cudaMemcpyHostToDevice, stream));
	}

	// ReSharper disable once CppMemberFunctionMayBeConst
	void set(const uint8_t val, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemsetAsync(mem, val, sizeof(T) * N, stream));
	}

	operator T *() const {
		return mem;
	}

	device::array <T> &operator= (const T *rhs) const {
		fromHost(rhs);
		return *this;
	}
};