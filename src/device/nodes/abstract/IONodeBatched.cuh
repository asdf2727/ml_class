#pragma once

#include "weightedNodeBatched.cuh"

namespace device {
	template <typename array_t>
	class IONodeBatched;
};

/* Implement:
 * void makeForwardGraph();
 * void makeForwardGraph();
 * virtual bool editInput(array_t &input);
 * virtual bool editOutput(array_t &output);
 * void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0);
 * void loadWeights(const std::vector <float> &weights);
 * std::vector <float> saveWeights() const;
 * void descend(const float step_size, const cudaStream_t stream)
 */

template <typename array_t>
class device::IONodeBatched : public device::weightedNodeBatched {
	public:
	virtual bool editInput(array_t &input);
	virtual bool editOutput(array_t &output);
};