#pragma once

#include <array>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cublas_v2.h>
#include <curand.h>

#include "mem/array.cuh"

#define cudaTry(function) cudaError_t err = function; if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err))
#define cublasTry(function) cublasStatus_t status = function; if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasGetStatusString(status))
#define calcBlocks(n, block_size) ((n + block_size - 1) / block_size)

// Will probably be unused, just an exercise to make a better kernel
template <typename T, size_t repeats>
__global__ void devFillArray_WTF(T* array, const size_t n, const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * repeats;
	const size_t stop = std::min(n, (idx + 1) * repeats);
	#pragma unroll
	for (size_t cnt = repeats; cnt > 0; cnt >>= 1) { // TODO see if start < stop is a better condition
		if (start + cnt <= stop) {
			#pragma unroll
			for (size_t pos = 0; pos < cnt; pos++) {
				array[start++] = value;
			}
		}
	}
}

template <typename T>
__global__ void devFillArray(T* array, const size_t n, const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		array[idx] = value;
	}
}

template <typename T>
__global__ void devFillMatrix(T* matrix, const size_t pitch, const size_t n, const size_t m, const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < m) {
		matrix[idx + pitch * idy] = value;
	}
}

template <typename T>
inline void cudaFillArray(T* array, const size_t n, const T value, cudaStream_t stream = cudaStreamDefault) {
	constexpr size_t block_size = 256;
	const size_t grid_size = calcBlocks(n, block_size);
	devFillArray<<<grid_size, block_size, 0, stream>>>(array, n, value);
}

template <typename T>
inline void cudaFillMatrix(T* mat, const size_t pitch, const size_t n, const size_t m, const T value, cudaStream_t stream = cudaStreamDefault) {
	constexpr dim3 block_size(16, 16);  // 16 x 16 threads per block (256 total)
	const dim3 grid_size(calcBlocks(n, block_size.x),
						calcBlocks(m, block_size.x));
	devFillMatrix<<<block_size, grid_size, 0, stream>>>(mat, pitch, n, m, value);
}

