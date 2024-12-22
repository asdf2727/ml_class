#include <iostream>

#include "device/networks/multiLayerPerceptron.cuh"
#include "device/nodes/activations/sigmoid.cuh"
#include "device/nodes/losses/binaryCrossEntropy.cuh"

int main() {
	const std::vector<device::multiLayerPerceptron::layerParams> params = {
		{50, device::linear},
		{10, device::sigmoid},
		{2, device::sigmoid},
	};
	device::multiLayerPerceptron mlp(params, device::BCE, 32, 0.01);
	return 0;
}
