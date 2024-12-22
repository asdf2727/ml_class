#pragma once

#include "../../link.cuh"
#include "../../lazy.cuh"
#include "../wrappers/graph.cuh"
#include "../wrappers/matrix.cuh"
#include "../neurons/neuronArrayBatched.cuh"

namespace device {
	struct lossFunction;
	class calculateLossBatched;
}

struct device::lossFunction {
	float (*val) (const float output, const float expected);
	float (*der) (const float output, const float expected);

	lossFunction (float (*val) (const float output, const float expected),
	              float (*der) (const float output, const float expected)) :
		val(val),
		der(der) {}

	bool operator== (const device::lossFunction &other) const {
		return val == other.val && der == other.der;
	}
};

namespace device {
	const lossFunction dummy_loss(nullptr, nullptr);
}

class device::calculateLossBatched {
	device::neuronArrayBatched &output = dummy_array_batched; // TODO fix; this is dirty
	link <device::matrix <float>> expected;

	void buildGraph (device::graph *&graph);
	lazy <device::graph> graph = lazy <device::graph>
			(std::bind(&calculateLossBatched::buildGraph, this, std::placeholders::_1));

public:
	const lossFunction loss = device::dummy_loss; // TODO fix this as well

	calculateLossBatched (device::neuronArrayBatched &output, device::matrix <float> &expected,
	                      const device::lossFunction &loss) :
		output(output),
		expected(expected),
		loss(loss) { assert(output.size == expected.X && output.batch_size == expected.Y); }

	calculateLossBatched () = default;

	calculateLossBatched (const calculateLossBatched &other) :
		output(other.output),
		expected(other.expected),
		loss(other.loss) {}
	calculateLossBatched &operator= (const calculateLossBatched &other) {
		if (this == &other) return *this;
		output = other.output;
		expected = other.expected;
		(lossFunction &)loss = other.loss;
		return *this;
	}
	calculateLossBatched (calculateLossBatched &&other) noexcept :
		output(other.output),
		expected(std::move(other.expected)),
		loss(other.loss) {}
	calculateLossBatched &operator= (calculateLossBatched &&other) noexcept {
		if (this == &other) return *this;
		output = other.output;
		expected = std::move(other.expected);
		(lossFunction &)loss = other.loss;
		return *this;
	}
	~calculateLossBatched() = default;

	void changeExpected (device::matrix <float> &new_expected);

	device::graph getGraph () const { return *graph; }
};
