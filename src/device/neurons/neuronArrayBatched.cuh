#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/matrix.cuh"

namespace device {
	struct neuronArrayBatched;
};

struct device::neuronArrayBatched {
	device::matrix <float> val;

	void buildDer(const device::matrix <float> *&der) const { der = new device::matrix <float>(val.X, val.Y); };
	lazy <device::matrix <float>, neuronArrayBatched> der;

	const size_t &size, batch_size;

	neuronArrayBatched (const size_t size, const size_t batch_size) :
	val(size + 1, batch_size), der(buildDer), size(val.X), batch_size(val.Y) {
		cudaFillMatrix <float>(val + val.X - 1, val.pitch, 1, batch_size, 1);
	}

	neuronArrayBatched &operator= (neuronArrayBatched &&other) noexcept {
		assert(size == other.size && batch_size == other.batch_size);
		val = std::move(other.val);
		der = std::move(other.der);
		return *this;
	}
};