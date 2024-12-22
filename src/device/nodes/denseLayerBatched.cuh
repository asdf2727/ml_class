#pragma once

#include "activateLayerBatched.cuh"
#include "linearLayerBatched.cuh"
#include "abstract/weightedNodeBatched.cuh"
#include "activations/all.cuh"

namespace device {
	class denseLayerBatched;
}

class device::denseLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
	device::linearLayerBatched mul;
	device::activateLayerBatched act;

public:
	const size_t &in_size, out_size, batch_size;

private:
	void buildForward (device::graph *&fwd) override;
	void buildBackward (device::graph *&back) override;
	void buildDescent (device::graph *&desc) override { desc = &mul.getDescent(); }

public:
	denseLayerBatched (device::neuronArrayBatched &input, device::neuronArrayBatched &output,
	                   const device::activationFunction &act) :
		IONodeBatched(input, output),
		mul(input, output),
		act(output, act),
		in_size(mul.in_size),
		out_size(mul.out_size),
		batch_size(mul.batch_size) {}

	void changeInput (device::neuronArrayBatched &input) override;
	void changeOutput (device::neuronArrayBatched &output) override;

	void changeStepSize (const float new_step_size) override;

	void resetWeights (const float mean, const float std_dev,
	                   const unsigned long long seed) override;

	void loadWeights (const std::vector <float> &weights) override { mul.loadWeights(weights); }
	[[nodiscard]] std::vector <float> saveWeights () const override { return mul.saveWeights(); }

	[[nodiscard]] device::activationType getType () const {
		return device::getActType(act.act);
	}
};