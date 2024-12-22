#pragma once

#include "../wrappers/matrix.cuh"
#include "../neurons/neuronArrayBatched.cuh"
#include "abstract/IONodeBatched.cuh"

namespace device {
	class linearLayerBatched;
}

class device::linearLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
public:
	const size_t &in_size, out_size, batch_size;

private:
	static constexpr float const1 = 1.0, const0 = 0.0;

	device::matrix <float> mul;

	void buildMultDer (device::matrix <float> *&mul_der) {
		mul_der = new device::matrix <float>(in_size + 1, out_size);
		mul_der->set(0x00);
	}
	lazy <device::matrix <float>> mul_der = lazy <device::matrix <float>>
			(std::bind(&linearLayerBatched::buildMultDer, this, std::placeholders::_1));

	void buildForward (device::graph *&fwd) override;
	void buildBackward (device::graph *&back) override;
	void buildDescent (device::graph *&desc) override;

public:
	linearLayerBatched (device::neuronArrayBatched &input, device::neuronArrayBatched &output) :
		IONodeBatched(input, output),
		in_size(this->input->size),
		out_size(this->output->size),
		batch_size(this->input->batch_size),
		mul(in_size, out_size) {}

	void resetWeights (const float mean, const float std_dev,
	                   const unsigned long long seed) override;
	void loadWeights (const std::vector <float> &weights) override;

	[[nodiscard]] std::vector <float> saveWeights () const override;
};