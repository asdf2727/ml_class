#pragma once

#include "../wrappers/matrix.cuh"
#include "../neurons/neuronArrayBatched.cuh"
#include "abstract/IONodeBatched.cuh"

namespace device {
	class linearLayerBatched;
}

class device::linearLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
	static constexpr float const1 = 1.0, const0 = 0.0;

	device::matrix <float> mul;

	lazy <device::matrix <float>> mul_der = lazy <device::matrix <float>>
	([this] (device::matrix <float> *&mul_der) {
		device::matrix <float> *temp = new device::matrix <float>(getInSize() + 1, getOutSize());
		mul_der = temp;
		mul_der->set(0x00);
	});

	void buildForward (device::graph *&fwd) override;
	void buildBackward (device::graph *&back) override;
	void buildDescent (device::graph *&desc) override;

public:
	linearLayerBatched (device::neuronArrayBatched &input, device::neuronArrayBatched &output) :
		IONodeBatched(input, output),
		mul(input.getSize() + 1, output.getSize()) {}

	linearLayerBatched (const device::linearLayerBatched &other) = delete;
	device::linearLayerBatched &operator= (const device::linearLayerBatched &rhs) = delete;
	linearLayerBatched (device::linearLayerBatched &&other) noexcept :
		IONodeBatched(std::move(other)),
		mul(std::move(other.mul)) {}
	device::linearLayerBatched &operator= (device::linearLayerBatched &&rhs) noexcept {
		if (this == &rhs) return *this;
		IONodeBatched::operator=(std::move(rhs));
		mul = std::move(rhs.mul);
		return *this;
	}
	~linearLayerBatched () override = default;

	void resetWeights (const float mean, const float std_dev,
	                   const unsigned long long seed) override;
	void loadWeights (const std::vector <float> &weights) override;

	[[nodiscard]] std::vector <float> saveWeights () const override;
};
