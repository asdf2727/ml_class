#pragma once

#include "../device.cuh"

namespace device {
	template <typename T>
	struct matrix;
}

template <typename T>
struct device::matrix {
	const size_t X, Y;
	const size_t pitch;

private:
	T *mem;

public:
	matrix (const size_t X, const size_t Y) : X(X), Y(Y), pitch(0) {
		cudaTry(cudaMallocPitch(&mem, (size_t*)&pitch, X * sizeof(T), Y));
	}

	matrix() : X(0), Y(0), pitch(0), mem(nullptr) {}

	void copy (const device::matrix <T> &arr, const cudaStream_t stream = cudaStreamLegacy);

	matrix (const device::matrix <T> &other) : matrix(other.X, other.Y) { copy(other); }
	device::matrix <T> &operator= (const device::matrix <T> &rhs) {
		copy(rhs);
		return *this;
	}
	matrix (device::matrix <T> &&other) noexcept : X(other.X), Y(other.Y), pitch(other.pitch),
	                                               mem(other.mem) { other.mem = nullptr; }
	device::matrix <T> &operator= (device::matrix <T> &&other) noexcept {
		assert(X == other.X && Y == other.Y);
		cudaTry(cudaFree(mem));
		mem = other.mem;
		*(size_t*)&pitch = other.pitch;
		other.mem = nullptr;
		return *this;
	}
	~matrix () { cudaTry(cudaFree(mem)); }

	void toHost (T *arr, const size_t arr_pitch,
	             const cudaStream_t stream = cudaStreamLegacy) const;
	void toHost (T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		toHost(arr, X * sizeof(T), stream);
	}

	void fromHost (const T *arr, const size_t arr_pitch,
	               const cudaStream_t stream = cudaStreamLegacy) const;
	void fromHost (const T *arr, const cudaStream_t stream = cudaStreamLegacy) const {
		fromHost(arr, X * sizeof(T), stream);
	}

	void set (const uint8_t val, const cudaStream_t stream = cudaStreamLegacy) const;

	void fill (const T val, const size_t begin_x, const size_t end_x, const size_t begin_y,
	           const size_t end_y, const cudaStream_t stream = cudaStreamLegacy);
	void fill (const T val, const cudaStream_t stream = cudaStreamLegacy) {
		fill(val, 0, X, 0, Y, stream);
	}

	operator T* () { return mem; }
	operator const T* () const { return mem; }

	device::matrix <T> &operator= (const T *rhs) {
		fromHost(rhs);
		return *this;
	}
	const device::matrix <T> &operator= (const T *rhs) const {
		fromHost(rhs);
		return *this;
	}
};

template <typename T>
void device::matrix <T>::copy (const device::matrix <T> &arr, const cudaStream_t stream) {
	assert(X == arr.X && Y == arr.Y);
	cudaTry(cudaMemcpy2DAsync(mem, pitch, arr.mem, arr.pitch, X * sizeof(T), Y,
	                          cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void device::matrix <T>::toHost (T *arr, const size_t arr_pitch, const cudaStream_t stream) const {
	cudaTry(cudaMemcpy2DAsync(arr, arr_pitch, mem, pitch, X * sizeof(T), Y, cudaMemcpyDeviceToHost,
	                          stream));
}

template <typename T>
void device::matrix <T>::fromHost (const T *arr, const size_t arr_pitch,
                                   const cudaStream_t stream) const {
	cudaTry(cudaMemcpy2DAsync(mem, pitch, arr, arr_pitch, X * sizeof(T), Y, cudaMemcpyHostToDevice,
	                          stream));
}

template <typename T>
void device::matrix <T>::set (const uint8_t val, const cudaStream_t stream) const {
	cudaTry(cudaMemset2DAsync(mem, pitch, val, sizeof(T) * X, Y, stream));
}

template <typename T>
__global__ void devFillMatrix (T *matrix, const size_t pitch, const size_t n, const size_t m,
                               const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < m) { matrix[idx + pitch * idy] = value; }
}
template <typename T>
void device::matrix <T>::fill (const T val, const size_t begin_x, const size_t end_x,
                               const size_t begin_y,
                               const size_t end_y, const cudaStream_t stream) {
	assert(begin_x < end_x && end_x <= X && begin_y < end_y && end_y <= Y);
	const size_t len_x = end_x - begin_x;
	const size_t len_y = end_y - begin_y;
	constexpr dim3 block_size(16, 16); // 16 x 16 threads per block (256 total)
	const dim3 grid_size(calcBlocks(len_x, block_size.x),
	                     calcBlocks(len_y, block_size.x));
	devFillMatrix<<<block_size, grid_size, 0, stream>>>(mem + (begin_x * sizeof(T))+ begin_y * pitch,
	                                                    pitch, len_x, len_y, val);
}
