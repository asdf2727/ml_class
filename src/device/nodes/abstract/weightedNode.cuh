#pragma once

#include <vector>
#include <iostream>
#include "node.cuh"

namespace device {
	class weightedNode;
};

/* Implement:
 * void makeForwardGraph();
 * void makeForwardGraph();
 * void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0);
 * void loadWeights(const std::vector <float> &weights);
 * std::vector <float> saveWeights() const;
 * void descend(const float step_size, const cudaStream_t stream)
 */

class device::weightedNode : public virtual device::node {
public:
	virtual void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0);

	virtual void loadWeights(const std::vector <float> &weights);
	virtual std::vector <float> saveWeights() const;

	void readWeights(std::istream &in) {
		size_t size;
		in.read(reinterpret_cast<char *>(&size), sizeof(size));
		std::vector <float> weights(size);
		in.read(reinterpret_cast <char*>(weights.data()), sizeof(float) * size);
		loadWeights(weights);
	}
	void writeWeights(std::ostream &out) const {
		const std::vector <float> weights(saveWeights());
		const size_t size = weights.size();
		out.write(reinterpret_cast<const char *>(&size), sizeof(size));
		out.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(float));
	}

	virtual void descend(const float step_size, const cudaStream_t stream);
};