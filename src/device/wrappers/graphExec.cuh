#pragma once

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

void device::graphExec::update (const device::graph &new_graph)  {
	if (data != nullptr) {
		cudaGraphExecUpdateResult updateResult;
		cudaGraphNode_t errorNode;
		cudaTry(cudaGraphExecUpdate(data, new_graph, &errorNode, &updateResult));
		if (updateResult != cudaGraphExecUpdateSuccess) {
			cudaTry(cudaGraphExecDestroy(data));
			data = nullptr;
		}
	}
	if (data == nullptr) {
		cudaTry(cudaGraphInstantiate(&data, new_graph, nullptr, nullptr, 0));
	}
}