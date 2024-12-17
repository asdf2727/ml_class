#pragma once

#include <array>

#include "../device.cuh"

namespace device {
	template <typename T>
	struct matrix;
};

template <typename T>
struct device::matrix {
	const size_t X, Y;
	const size_t pitch;

private:
	T *mem;

public:
	matrix(const size_t X, const size_t Y) : X(X), Y(Y), pitch(0) {
		cudaTry(cudaMallocPitch(&mem, &pitch, X * sizeof(T), Y));
	}

	~matrix() {
		cudaTry(cudaFree(mem));
	}

	void toHost(T *arr, const size_t arr_pitch, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemcpy2DAsync(arr, arr_pitch, mem, mem.pitch, X * sizeof(T), Y, cudaMemcpyDeviceToHost, stream));
	}
	void toHost(T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		cudaTry(cudaMemcpy2DAsync(arr, X * sizeof(T), mem, mem.pitch, X * sizeof(T), Y, cudaMemcpyDeviceToHost, stream));
	}
	void fromHost(const T *arr, const size_t arr_pitch, const cudaStream_t stream = cudaStreamLegacy) {
		cudaTry(cudaMemcpy2DAsync(mem, pitch, arr, arr_pitch, X * sizeof(T), Y, cudaMemcpyHostToDevice, stream));
	}
	void fromHost(const T *arr, const cudaStream_t stream = cudaStreamLegacy) {
		cudaTry(cudaMemcpy2DAsync(mem, pitch, arr, X * sizeof(T), X * sizeof(T), Y, cudaMemcpyHostToDevice, stream));
	}

	void copy(const device::matrix<T> &arr, const cudaStream_t stream = cudaStreamLegacy) {
		assert(X == arr.X && Y == arr.Y);
		cudaTry(cudaMemcpy2DAsync(mem, pitch, arr.mem, arr.pitch, X * sizeof(T), Y, cudaMemcpyDeviceToDevice, stream));
	}

	// ReSharper disable once CppMemberFunctionMayBeConst
	void set(const uint8_t val, const cudaStream_t stream = cudaStreamLegacy) {
		cudaTry(cudaMemset2DAsync(mem, pitch, val, sizeof(T) * X, Y, stream));
	}

	operator T *() const {
		return mem;
	}

	device::matrix  <T> &operator= (const T *rhs) {
		fromHost(rhs);
		return *this;
	}
	device::matrix  <T> &operator= (const device::matrix <T> &rhs) {
		copy(rhs);
		return *this;
	}
};
