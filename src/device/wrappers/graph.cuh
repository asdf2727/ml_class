#pragma once

#include "../device.cuh"

namespace device {
	class graph;
};

class device::graph {
	cudaGraph_t data;

public:
	graph() : data(nullptr) {
		cudaTry(cudaGraphCreate(&data, 0));
	}

	~graph() {
		cudaTry(cudaGraphDestroy(data));
	}

	friend ::operator cudaGraph_t *(graph *rhs) {
		return &rhs->data;
	}

	operator cudaGraph_t &() {
		return data;
	}
	operator cudaGraph_t () const {
		return data;
	}
};