#pragma once

#include <cmath>
#include <cassert>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

inline void cudaTry (const cudaError_t err) {
	if (err != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(err));
}
inline void cublasTry (const cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(cublasGetStatusString(status));
}

constexpr size_t calcBlocks (const size_t n, const size_t block_size) { return (n + block_size - 1) / block_size; }

// Will probably be unused, just an exercise to make a better kernel
template <typename T, size_t repeats> // preferably power of 2
__global__ void devFillArray_WTF (T *array, const size_t n, const T value) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * repeats;
	const size_t stop = std::min(n, (idx + 1) * repeats);
	#pragma unroll
	for (size_t cnt = repeats; cnt > 0; cnt >>= 1) {
		// TODO see if start < stop is a better condition
		if (start + cnt <= stop) {
			#pragma unroll
			for (size_t pos = 0; pos < cnt; pos++) { array[start++] = value; }
		}
	}
}