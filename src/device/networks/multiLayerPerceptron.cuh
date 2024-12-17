#pragma once

#include <vector>

#include "../mem/neuronArrayBatched.cuh"
#include "../nodes/activations/sigmoidLayerBatched.cuh"
#include "../nodes/denseLayerBatched.cuh"

namespace device {
	class multiLayerPerceptron;
};

class device::multiLayerPerceptron : device::node {
	std::vector <device::neuronArrayBatched> arrays;

	template <typename T>
	struct block : device::node {
		device::linearLayerBatched multiply;
		T activate;

		block(const device::neuronArrayBatched &input, const device::neuronArrayBatched &output) : multiply(input, output), activate(output) {

		}
	};
	std::vector <std::pair <device::linearLayerBatched, device::sigActivateBatched>> blocks;

	cudaGraph_t makeForwardGraph();
	cudaGraph_t makeBackwardGraph();
public:
	multiLayerPerceptron(std::vector <size_t> &sizes, const size_t batch_size) {
		arrays.emplace_back(sizes[0], batch_size);
		for (size_t i = 1; i < sizes.size() - 1; i++) {
			arrays.emplace_back(sizes[i], batch_size);
			blocks.emplace_back(new arrays[i - 1], arrays[i], arrays[i]);
			activations.emplace_back(arrays[i]);
		}
	}
};