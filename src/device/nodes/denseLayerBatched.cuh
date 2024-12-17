#pragma once

#include "linearLayerBatched.cuh"
#include "activations/sigmoidLayerBatched.cuh"

namespace device {
	template <typename activation>
	class denseLayerBatched;
};

template <typename activation>
class device::denseLayerBatched : public device::weightedNode {
	device::linearLayerBatched mul;
	activation act;

public:
	const size_t &in_size, out_size, batch_size;

	inline void makeForwardGraph();
	inline void makeBackwardGraph();

public:
	denseLayerBatched(device::neuronArrayBatched &input, device::neuronArrayBatched &output) :
	mul(input, output), act(output), in_size(mul.in_size), out_size(mul.out_size), batch_size(mul.batch_size) { }

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

template <typename activation>
void device::denseLayerBatched <activation>::makeForwardGraph () {
	cudaGraphCreate(&fwd, 0);

	cudaGraphNode_t mul_node, act_node;
	cudaGraphAddChildGraphNode(&mul_node, fwd, nullptr, 0, mul.getForwardGraph());
	cudaGraphAddChildGraphNode(&act_node, fwd, *mul_node, 1, act.getForwardGraph());
}

template <typename activation>
void device::denseLayerBatched <activation>::makeBackwardGraph () {
	cudaGraphCreate(&back, 0);

	cudaGraphNode_t mul_node, act_node;
	cudaGraphAddChildGraphNode(&mul_node, back, nullptr, 0, mul.getBackwardGraph());
	cudaGraphAddChildGraphNode(&act_node, fwd, *mul_node, 1, act.getBackwardGraph());

	cudaGraphNodeParams
}