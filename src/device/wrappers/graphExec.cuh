#pragma once

#include "graph.cuh"
#include "../device.cuh"

namespace device {
	class graphExec;
}

class device::graphExec {
	cudaGraphExec_t data;

public:
	graphExec () : data(nullptr) {}

	~graphExec () { cudaTry(cudaGraphExecDestroy(data)); }

	operator cudaGraphExec_t () const { return data; }
	operator cudaGraphExec_t& () { return data; }
	const cudaGraphExec_t *operator* () const { return &data; }
	cudaGraphExec_t *operator* () { return &data; }

	void update (const device::graph &new_graph);
};