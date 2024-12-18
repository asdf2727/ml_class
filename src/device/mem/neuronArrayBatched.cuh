#pragma once

#include "../device.cuh"
#include "matrix.cuh"

namespace device {
	struct neuronArrayBatched;
};

struct device::neuronArrayBatched {
	const size_t size, batch_size;
	device::matrix <float> val;
	device::matrix <float> der;

	neuronArrayBatched (const size_t size, const size_t batch_size) :
	size(size), batch_size(batch_size), val(size + 1, batch_size), der(size, batch_size) {
		cudaFillMatrix <float>(val, val.pitch, 1, batch_size, 1);
	}

	neuronArrayBatched &operator= (neuronArrayBatched &&other) noexcept {
		assert(size == other.size && batch_size == other.batch_size);
		val = std::move(other.val);
		der = std::move(other.der);
		return *this;
	}
};