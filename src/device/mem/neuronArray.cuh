#pragma once

#include "../device.cuh"
#include "array.cuh"

namespace device {
	struct neuronArray;
};

struct device::neuronArray {
	const size_t size;
	device::array <float> val;
	device::array <float> der;

	explicit neuronArray(const size_t size) :
	size(size), val(size + 1), der(size) {
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