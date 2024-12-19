#pragma once

#include "weightedNodeBatched.cuh"
#include "../../../link.cuh"

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
protected:
	link <array_t> input;
	link <array_t> output;

public:
	IONodeBatched(array_t &input, array_t &output) : input(input), output(output) {
		assert(this->input.get().batch_size == this->output.get().batch_size);
	}

	virtual void changeInput(array_t &input) {
		this.input = input;
		fwd.invalidate();
		back.invalidate();
	}
	virtual void changeOutput(array_t &output) {
		this.output = output;
		fwd.invalidate();
		back.invalidate();
	}
};