#pragma once

#include "../device.cuh"

namespace device {
	class graph;
}

class device::graph {
	cudaGraph_t data;

public:
	graph () : data(nullptr) { cudaTry(cudaGraphCreate(&data, 0)); }

	~graph () { cudaTry(cudaGraphDestroy(data)); }

	operator cudaGraph_t () const { return data; }
	operator cudaGraph_t& () { return data; }
	const cudaGraph_t *operator* () const { return &data; }
	cudaGraph_t *operator* () { return &data; }
};
