#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/array.cuh"

namespace device {
	struct neuronArray;
};

struct device::neuronArray {
	device::array <float> val;

	void buildDer(const device::array <float> *&der) const { der = new device::array <float>(val.N); };
	lazy <device::array <float>, neuronArray> der;

	const size_t &size;

	explicit neuronArray(const size_t size) :
	val(size + 1), der(buildDer), size(val.N) {
		static constexpr float just_a_1 = 1;
		cudaTry(cudaMemcpy(val + size, &just_a_1, 1, cudaMemcpyHostToDevice));
	}

	neuronArray &operator= (neuronArray &&other) noexcept {
		assert(size == other.size);
		val = std::move(other.val);
		der = std::move(other.der);
		return *this;
	}
};