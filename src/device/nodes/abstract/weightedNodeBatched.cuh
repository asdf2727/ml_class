#pragma once

#include "nodeBatched.cuh"
#include "weightedNode.cuh"

namespace device {
	class weightedNodeBatched;
};

/* Implement:
 * void makeForwardNode();
 * void makeBackwardNode();
 * void remakeGraphBatches();
 * void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0);
 * void loadWeights(const std::vector <float> &weights);
 * std::vector <float> saveWeights() const;
 * void descend(const float step_size, const cudaStream_t stream)
 */

class device::weightedNodeBatched : device::weightedNode, device::nodeBatched {
};