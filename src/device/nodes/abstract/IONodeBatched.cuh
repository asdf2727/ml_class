#pragma once

#include "weightedNodeBatched.cuh"
#include "../../../link.cuh"

namespace device {
	template <typename array_t>
	class IONodeBatched;
}

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
	IONodeBatched (array_t &input, array_t &output) :
		weightedNodeBatched(input.getBatchSize()),
		input(input),
		output(output) { assert(this->input->getBatchSize() == this->output->getBatchSize()); }

	IONodeBatched (const IONodeBatched &other) = delete;
	IONodeBatched &operator= (const IONodeBatched &other) = delete;
	IONodeBatched (IONodeBatched &&other) noexcept :
		weightedNodeBatched(std::move(other)),
		input(std::move(other.input)),
		output(std::move(other.output)) {}
	IONodeBatched &operator= (IONodeBatched &&rhs) noexcept {
		if (this == &rhs) return *this;
		weightedNodeBatched::operator=(std::move(rhs));
		input = std::move(rhs.input);
		output = std::move(rhs.output);
		fwd = std::move(rhs.fwd);
		back = std::move(rhs.back);
		return *this;
	}
	~IONodeBatched () override = default;

	size_t getInSize() const { return input->getSize(); }
	size_t getOutSize() const { return output->getSize(); }

	virtual void changeInput (array_t &input) {
		assert(getBatchSize() == input.getBatchSize());
		assert(getInSize() == input.getSize());
		this->input = input;
		fwd.invalidate();
		back.invalidate();
	}
	virtual void changeOutput (array_t &output) {
		assert(getBatchSize() == output.getBatchSize());
		assert(getOutSize() == output.getSize());
		this->output = output;
		fwd.invalidate();
		back.invalidate();
	}
};
