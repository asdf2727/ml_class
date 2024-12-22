#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/matrix.cuh"

namespace device {
	struct neuronArrayBatched;
}

struct device::neuronArrayBatched {
	device::matrix <float> val;

	lazy <device::matrix <float>> der = lazy <device::matrix <float>>
			([this](device::matrix <float> *&der) { der = new device::matrix <float>(val.X, val.Y); });

	neuronArrayBatched (const size_t &size, const size_t &batch_size) :
		val(size + 1, batch_size) { val.fill(1, val.X - 1, val.X, 0, val.Y); }

	neuronArrayBatched () = default;

	neuronArrayBatched (const neuronArrayBatched &other) :
		val(other.val) {
		if (other.der.isValid()) *der = *other.der;
	}
	neuronArrayBatched &operator= (const neuronArrayBatched &rhs) {
		if (this == &rhs) return *this;
		val = rhs.val;
		if (rhs.der.isValid()) *der = *rhs.der;
		return *this;
	}
	neuronArrayBatched (neuronArrayBatched &&other) noexcept :
		val(std::move(other.val)),
		der(std::move(other.der)) {}
	neuronArrayBatched &operator= (neuronArrayBatched &&rhs) noexcept {
		if (this == &rhs) return *this;
		val = std::move(rhs.val);
		der = std::move(rhs.der);
		return *this;
	}
	~neuronArrayBatched () = default;

	size_t getSize() const { return val.X - 1; }
	size_t getBatchSize() const { return val.Y; }
};