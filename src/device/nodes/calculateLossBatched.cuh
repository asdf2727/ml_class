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

	lossFunction () : val(nullptr), der(nullptr) {}

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
	link <device::neuronArrayBatched> output;
	link <device::matrix <float>> expected;

	void buildGraph (device::graph *&graph);
	lazy <device::graph> graph = lazy <device::graph>
			([this] (device::graph *&graph) { buildGraph(graph); });

public:
	const lossFunction loss;

	calculateLossBatched (device::neuronArrayBatched &output, device::matrix <float> &expected,
	                      const device::lossFunction &loss) :
		output(output),
		expected(expected),
		loss(loss) { assert(output.getSize() == expected.X && output.getBatchSize() == expected.Y); }

	calculateLossBatched () :
		loss(nullptr, nullptr) {}

	calculateLossBatched (const calculateLossBatched &other) = delete;
	calculateLossBatched &operator= (const calculateLossBatched &other) = delete;
	calculateLossBatched (calculateLossBatched &&other) noexcept :
		output(std::move(other.output)),
		expected(std::move(other.expected)),
		loss(std::move(other.loss)) {}
	calculateLossBatched &operator= (calculateLossBatched &&other) noexcept {
		if (this == &other) return *this;
		output = std::move(other.output);
		expected = std::move(other.expected);
		(lossFunction&)loss = std::move(other.loss);
		return *this;
	}
	~calculateLossBatched () = default;

	void changeExpected (device::matrix <float> &new_expected);

	device::graph getGraph () const { return *graph; }
};
