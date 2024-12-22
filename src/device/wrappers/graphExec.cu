#include "graphExec.cuh"

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