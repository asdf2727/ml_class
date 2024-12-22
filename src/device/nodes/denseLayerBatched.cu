#include "denseLayerBatched.cuh"

void device::denseLayerBatched::buildForward (device::graph *&fwd) {
	cudaGraphCreate((cudaGraph_t*)fwd, 0);

	cudaGraphNode_t mul_node, act_node;
	cudaGraphAddChildGraphNode(&mul_node, *fwd, nullptr, 0, mul.getForward());
	if (getActType(act.act) != ACT_NONE) {
		cudaGraphAddChildGraphNode(&act_node, *fwd, &mul_node, 1, act.getForward());
	}
}
void device::denseLayerBatched::buildBackward (device::graph *&back) {
	cudaGraphCreate((cudaGraph_t*)back, 0);

	cudaGraphNode_t mul_node, act_node;
	if (getActType(act.act) != ACT_NONE) {
		cudaGraphAddChildGraphNode(&act_node, *back, nullptr, 0, act.getBackward());
	}
	cudaGraphAddChildGraphNode(&mul_node, *back, &mul_node, 1, mul.getBackward());
}

void device::denseLayerBatched::changeInput (device::neuronArrayBatched &input) {
	mul.changeInput(input);
	IONodeBatched::changeInput(input);
}
void device::denseLayerBatched::changeOutput (device::neuronArrayBatched &output) {
	IONodeBatched::changeOutput(output);
	mul.changeOutput(output);
	act.changeData(output);
}

void device::denseLayerBatched::resetWeights (const float mean, const float std_dev,
				   const unsigned long long seed) {
	mul.resetWeights(mean, std_dev, seed);
}

void device::denseLayerBatched::changeStepSize (const float new_step_size) {
	IONodeBatched::changeStepSize(new_step_size);
	mul.changeStepSize(step_size);
}