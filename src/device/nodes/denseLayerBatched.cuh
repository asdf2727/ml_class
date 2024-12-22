#pragma once

#include "activateLayerBatched.cuh"
#include "linearLayerBatched.cuh"
#include "abstract/weightedNodeBatched.cuh"
#include "activations/common.cuh"

namespace device {
	class denseLayerBatched;
}

class device::denseLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
	device::linearLayerBatched mul;
	device::activateLayerBatched act;

private:
	void buildForward (device::graph *&fwd) override;
	void buildBackward (device::graph *&back) override;
	void buildDescent (device::graph *&desc) override { desc = &mul.getDescent(); }

public:
	denseLayerBatched (device::neuronArrayBatched &input, device::neuronArrayBatched &output,
	                   const device::activationFunction &act) :
		IONodeBatched(input, output),
		mul(input, output),
		act(output, act) {}

	denseLayerBatched (const denseLayerBatched &other) = delete;
	denseLayerBatched &operator= (const denseLayerBatched &other) = delete;
	denseLayerBatched (denseLayerBatched &&other) noexcept :
		IONodeBatched(std::move(other)),
		mul(std::move(other.mul)),
		act(std::move(other.act)) {}
	denseLayerBatched &operator= (denseLayerBatched &&rhs) noexcept {
		if (this != &rhs) return *this;
		IONodeBatched::operator=(std::move(rhs));
		mul = std::move(rhs.mul);
		act = std::move(rhs.act);
		return *this;
	}

	void changeInput (device::neuronArrayBatched &input);
	void changeOutput (device::neuronArrayBatched &output);

	void changeStepSize (const float new_step_size) override;

	void resetWeights (const float mean, const float std_dev,
	                   const unsigned long long seed) override;

	void loadWeights (const std::vector <float> &weights) override { mul.loadWeights(weights); }
	[[nodiscard]] std::vector <float> saveWeights () const override { return mul.saveWeights(); }

	[[nodiscard]] device::activationType getType () const { return device::getActType(act.act); }
};
