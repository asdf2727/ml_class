#pragma once

namespace device {
	class node;
};

/* Implement:
 * void makeForwardGraph();
 * void makeBackwardGraph();
 */

class device::node {
protected:
	cudaGraph_t fwd = nullptr, back = nullptr;
	virtual void makeForwardGraph();
	virtual void makeBackwardGraph();
	void invalidateGraphs() {
		if (fwd != nullptr) {
			cudaGraphDestroy(fwd);
			fwd = nullptr;
		}
		if (back != nullptr) {
			cudaGraphDestroy(back);
			back = nullptr;
		}
	}

public:
	virtual ~node() {
		invalidateGraphs();
	}

	cudaGraph_t getForwardGraph() {
		if (fwd == nullptr) makeForwardGraph();
		return fwd;
	}
	cudaGraph_t getBackwardGraph() {
		if (back == nullptr) makeBackwardGraph();
		return back;
	}

	virtual void descend(const float step_size, const cudaStream_t stream) { }
	cudaStream_t descend(const float step_size) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		descend(step_size, stream);
		return stream;
	}
};