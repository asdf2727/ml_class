#pragma once

#include <vector>
#include <iostream>
#include "node.cuh"

namespace device {
	class weightedNode;
}

/* Implement:
 * void makeForwardGraph();
 * void makeForwardGraph();
 * void makeDescentGraph();
 * void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0);
 * void loadWeights(const std::vector <float> &weights);
 * std::vector <float> saveWeights() const;
 * void descend(const float step_size, const cudaStream_t stream)
 */

class device::weightedNode : public virtual device::node {
protected:
	virtual void buildDescent (device::graph *&desc) = 0;
	lazy <device::graph> desc = lazy <device::graph>
			(std::bind(&weightedNode::buildDescent, this, std::placeholders::_1));

	float step_size = 0;

public:
	virtual void resetWeights (const float mean, const float std_dev,
	                           const unsigned long long seed) = 0;
	virtual void loadWeights (const std::vector <float> &weights) = 0;
	[[nodiscard]] virtual std::vector <float> saveWeights () const = 0;

	void readWeights (std::istream &in);
	void writeWeights (std::ostream &out) const;

	virtual void changeStepSize (const float new_step_size);

	device::graph &getDescent () { return *desc; }
};