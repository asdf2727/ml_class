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
	cudaGraphNode_t mul_node = nullptr, act_node = nullptr;

public:
	const size_t &in_size, out_size, batch_size;

	void makeForwardGraph() override;
	void makeBackwardGraph() override;

public:
	denseLayerBatched(device::neuronArrayBatched &input, device::neuronArrayBatched &output, const device::activationFunction act) :
	mul(input, output), act(output, act), in_size(mul.in_size), out_size(mul.out_size), batch_size(mul.batch_size) { }

	bool editInput(device::neuronArrayBatched &input) {
		if (mul.editInput(input)) {
			invalidateGraphs();
			return true;
		}
		return false;
	}
	bool editOutput(device::neuronArrayBatched &output) {
		if (mul.editOutput(output) || act.editData(output)) {
			invalidateGraphs();
			return true;
		}
		return false;
	}

	void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0) {
		mul.resetWeights(mean, std_dev, seed);
	}

	void loadWeights(const std::vector <float> &weights) {
		mul.loadWeights(weights);
	}
	std::vector <float> saveWeights() const {
		return mul.saveWeights();
	}

	void descend(const float step_size, const cudaStream_t stream) {
		mul.descend(step_size, stream);
	}
};

void device::denseLayerBatched::makeForwardGraph () {
	cudaGraphCreate(&fwd, 0);

	cudaGraphAddChildGraphNode(&mul_node, fwd, nullptr, 0, mul.getForwardGraph());
	cudaGraphAddChildGraphNode(&act_node, fwd, &mul_node, 1, act.getForwardGraph());
}

void device::denseLayerBatched::makeBackwardGraph () {
	cudaGraphCreate(&back, 0);

	cudaGraphAddChildGraphNode(&mul_node, back, nullptr, 0, mul.getBackwardGraph());
	cudaGraphAddChildGraphNode(&act_node, fwd, &mul_node, 1, act.getBackwardGraph());
}