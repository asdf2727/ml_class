#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/matrix.cuh"

namespace device {
	struct neuronArrayBatched;
}

struct device::neuronArrayBatched {
	device::matrix <float> val;

private:
	void buildDer (device::matrix <float> *&der) {
		der = new device::matrix <float>(val.X, val.Y);
	}

public:
	lazy <device::matrix <float>> der = lazy <device::matrix <float>>
			(std::bind(&neuronArrayBatched::buildDer, this, std::placeholders::_1));

	const size_t &size = val.X, batch_size = val.Y;

	neuronArrayBatched (const size_t &size, const size_t &batch_size) :
		val(size + 1, batch_size) { val.fill(1, val.X - 1, val.X, 0, val.Y); }

	neuronArrayBatched ();

	neuronArrayBatched (const neuronArrayBatched &other) :
		val(other.val) {
		if (other.der.isValid()) *der = *other.der;
	}
	neuronArrayBatched &operator= (const neuronArrayBatched &rhs) {
		assert(size == rhs.size && batch_size == rhs.batch_size);
		val = rhs.val;
		if (rhs.der.isValid()) *der = *rhs.der;
		return *this;
	}
	neuronArrayBatched (neuronArrayBatched &&other) noexcept :
		val(std::move(other.val)),
		der(std::move(other.der)) {}
	neuronArrayBatched &operator= (neuronArrayBatched &&rhs) noexcept {
		assert(size == rhs.size && batch_size == rhs.batch_size);
		val = std::move(rhs.val);
		der = std::move(rhs.der);
		return *this;
	}
	~neuronArrayBatched () = default;
};

namespace device {
	inline neuronArrayBatched dummy_array_batched (0, 0);
}