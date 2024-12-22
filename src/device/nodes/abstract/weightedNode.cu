#include "weightedNode.cuh"

#include "../../../binaryIO.cuh"

void device::weightedNode::readWeights (std::istream &in) {
	const size_t size = read <size_t>(in);
	std::vector <float> weights(size);
	read(in, weights.data(), size);
	loadWeights(weights);
}
void device::weightedNode::writeWeights (std::ostream &out) const {
	const std::vector <float> weights(saveWeights());
	const size_t size = weights.size();
	write(out, &size);
	write(out, weights.data(), size);
}

void device::weightedNode::changeStepSize (const float new_step_size) {
	step_size = new_step_size;
	desc.invalidate();
}