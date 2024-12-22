#pragma once

#include "../../lazy.cuh"
#include "../device.cuh"
#include "../wrappers/array.cuh"

namespace device {
	struct neuronArray;
}

struct device::neuronArray {
	device::array <float> val;

private:
	void buildDer (device::array <float> *&der) { der = new device::array <float>(val.N); }

public:
	lazy <device::array <float>> der = lazy <device::array <float>>
			(std::bind(&neuronArray::buildDer, this, std::placeholders::_1));

	const size_t &size = val.N;

	explicit neuronArray (const size_t size) :
		val(size + 1) {
		static constexpr float just_a_1 = 1;
		cudaTry(cudaMemcpy(val + size, &just_a_1, 1, cudaMemcpyHostToDevice));
	}

	neuronArray();

	neuronArray (const neuronArray &other) :
		val(other.val) { if (other.der.isValid()) *der = *other.der; }
	neuronArray &operator= (const neuronArray &rhs) {
		assert(size == rhs.size);
		val = rhs.val;
		if (rhs.der.isValid()) *der = *rhs.der;
		return *this;
	}
	neuronArray (neuronArray &&other) noexcept :
		val(std::move(other.val)),
		der(std::move(other.der)) {}
	neuronArray &operator= (neuronArray &&rhs) noexcept {
		assert(size == rhs.size);
		val = std::move(rhs.val);
		der = std::move(rhs.der);
		return *this;
	}
	~neuronArray () = default;
};

namespace device {
	inline neuronArray dummy_array (0);
}