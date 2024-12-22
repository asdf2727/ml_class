#pragma once

#include "../device.cuh"

namespace device {
	template <typename T>
	struct array;
}

template <typename T>
struct device::array {
	const size_t N;

private:
	T *mem;

public:
	explicit array (const size_t N) : N(N) { cudaTry(cudaMalloc(&mem, sizeof(T) * N)); }

	array () : N(0), mem(nullptr) {}

	void copy (const device::array <T> &arr, const cudaStream_t stream = cudaStreamLegacy);

	array (const device::array <T> &other) :
		array(other.N) { copy(other); }
	device::array <T> &operator= (const device::array <T> &rhs) {
		if (this == &rhs) return *this;
		copy(rhs);
		return *this;
	}
	array (device::array <T> &&other) noexcept :
		N(other.N),
		mem(other.mem) { other.mem = nullptr; }
	device::array <T> &operator= (device::array <T> &&rhs) noexcept {
		if (this == &rhs) return *this;
		cudaTry(cudaFree(mem));
		mem = rhs.mem;
		(size_t &)N = rhs.N;
		rhs.mem = nullptr;
		return *this;
	}
	~array () { cudaTry(cudaFree(mem)); }

	void toHost (T *arr, const cudaStream_t stream = cudaStreamLegacy) const;

	void fromHost (const T *arr, const cudaStream_t stream = cudaStreamLegacy) const;

	// ReSharper disable once CppMemberFunctionMayBeConst
	void set (const uint8_t val, const cudaStream_t stream = cudaStreamLegacy) const;

	void fill (const T val, const size_t begin, const size_t end,
	           const cudaStream_t stream = cudaStreamLegacy);
	void fill (const T val, const cudaStream_t stream = cudaStreamLegacy) {
		fill(val, 0, N, stream);
	}

	operator T* () { return mem; }
	operator const T* () const { return mem; }

	device::array <T> &operator= (const T *rhs) {
		fromHost(rhs);
		return *this;
	}
	const device::array <T> &operator= (const T *rhs) const {
		fromHost(rhs);
		return *this;
	}
};

template <typename T>
void device::array <T>::copy (const device::array <T> &arr, const cudaStream_t stream) {
	assert(N == arr.N);
	cudaTry(cudaMemcpyAsync(mem, arr.mem, sizeof(T) * N, cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void device::array <T>::toHost (T *arr, const cudaStream_t stream) const {
	cudaTry(cudaMemcpyAsync(arr, mem, sizeof(T) * N, cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void device::array <T>::fromHost (const T *arr, const cudaStream_t stream) const {
	cudaTry(cudaMemcpyAsync(mem, arr, sizeof(T) * N, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void device::array <T>::set (const uint8_t val, const cudaStream_t stream) const {
	cudaTry(cudaMemsetAsync(mem, val, sizeof(T) * N, stream));
}

template <typename T>
__global__ void devFillArray (T *array, const size_t n, const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) { array[idx] = value; }
}
template <typename T>
void device::array <T>::fill (const T val, const size_t begin, const size_t end,
                              const cudaStream_t stream) {
	assert(begin < end && end <= N);
	const size_t len = end - begin;
	constexpr size_t block_size = 256;
	const size_t grid_size = calcBlocks(len, block_size);
	devFillArray<<<block_size, grid_size, 0, stream>>>(mem + begin * sizeof(T), len, val);
}
