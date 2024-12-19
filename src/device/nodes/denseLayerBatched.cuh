#pragma once

#include "activateLayerBatched.cuh"
#include "linearLayerBatched.cuh"
#include "abstract/weightedNodeBatched.cuh"

namespace device {
	class denseLayerBatched;
};

class device::denseLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
	device::linearLayerBatched mul;
	device::activateLayerBatched act;

public:
	const size_t &in_size, out_size, batch_size;

	void buildForward (const device::graph *&fwd) const override;
	void buildBackward (const device::graph *&back) const override;
	void buildDescent (const device::graph *&desc) const override;

public:
	denseLayerBatched(device::neuronArrayBatched &input, device::neuronArrayBatched &output, const device::activationFunction act) :
	IONodeBatched(input, output), mul(input, output), act(output, act), in_size(mul.in_size), out_size(mul.out_size), batch_size(mul.batch_size) { }

	void changeInput(device::neuronArrayBatched &input) override {
		mul.changeInput(input);
		IONodeBatched::changeInput(input);
	}
	void changeOutput(device::neuronArrayBatched &output) override {
		mul.changeOutput(output);
		act.changeData(output);
		IONodeBatched::changeInput(output);
	}

	void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0) override {
		mul.resetWeights(mean, std_dev, seed);
	}

	void loadWeights(const std::vector <float> &weights) override {
		mul.loadWeights(weights);
	}
	std::vector <float> saveWeights() const override {
		return mul.saveWeights();
	}

	void changeStepSize(const float new_step_size) override {
		mul.changeStepSize(new_step_size);
		desc.invalidate();
	}
};

void device::denseLayerBatched::buildForward (const device::graph *&fwd) const {
	cudaGraphCreate((cudaGraph_t *)fwd, 0);

	cudaGraphNode_t mul_node, act_node;
	cudaGraphAddChildGraphNode(&mul_node, *fwd, nullptr, 0, mul.getForward());
	cudaGraphAddChildGraphNode(&act_node, *fwd, &mul_node, 1, act.getForward());
}
void device::denseLayerBatched::buildBackward (const device::graph *&back) const {
	cudaGraphCreate((cudaGraph_t *)back, 0);

	cudaGraphNode_t mul_node, act_node;
	cudaGraphAddChildGraphNode(&act_node, *back, nullptr, 0, act.getBackward());
	cudaGraphAddChildGraphNode(&mul_node, *back, &mul_node, 1, mul.getBackward());
}
void device::denseLayerBatched::buildDescent (const device::graph *&desc) const {
	desc = &mul.getDescent();
}