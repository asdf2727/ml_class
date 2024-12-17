#pragma once

namespace device {
	class node;
};

/* Implement:
 * void makeForwardNode();
 * void makeBackwardNode();
 */

class device::node {
protected:
	cudaGraph_t fwd = nullptr, back = nullptr;
	void makeForwardNode();
	void makeBackwardNode();

public:
	~node() {
		if (fwd != nullptr) cudaGraphDestroy(fwd);
		if (back != nullptr) cudaGraphDestroy(back);
	}

	cudaGraph_t getForwardGraph() {
		if (fwd == nullptr) makeForwardNode();
		return fwd;
	}
	cudaGraph_t getBackwardGraph() {
		if (back == nullptr) makeBackwardNode();
		return back;
	}

	void descend(const float step_size, const cudaStream_t stream) { }
	cudaStream_t descend(const float step_size) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		descend(step_size, stream);
		return stream;
	}
};