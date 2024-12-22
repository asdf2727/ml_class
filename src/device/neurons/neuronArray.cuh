#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/array.cuh"

namespace device {
	struct neuronArray;
}

struct device::neuronArray {
	device::array <float> val;

	lazy <device::array <float>> der = lazy <device::array <float>>
			([this](device::array <float> *&der) { der = new device::array <float>(val.N); });


	explicit neuronArray (const size_t size) :
		val(size + 1) {
		static constexpr float just_a_1 = 1;
		cudaTry(cudaMemcpy(val + size, &just_a_1, 1, cudaMemcpyHostToDevice));
	}

	neuronArray() = default;

	neuronArray (const neuronArray &other) :
		val(other.val) { if (other.der.isValid()) *der = *other.der; }
	neuronArray &operator= (const neuronArray &rhs) {
		if (this == &rhs) return *this;
		val = rhs.val;
		if (rhs.der.isValid()) *der = *rhs.der;
		return *this;
	}
	neuronArray (neuronArray &&other) noexcept :
		val(std::move(other.val)),
		der(std::move(other.der)) {}
	neuronArray &operator= (neuronArray &&rhs) noexcept {
		if (this == &rhs) return *this;
		val = std::move(rhs.val);
		der = std::move(rhs.der);
		return *this;
	}
	~neuronArray () = default;

	size_t getSize() const { return val.N - 1; }
};