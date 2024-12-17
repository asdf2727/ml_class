#pragma once

#include "node.cuh"

namespace device {
	class nodeBatched;
};

/* Implement:
 * void makeForwardNode();
 * void makeBackwardNode();
 * void remakeGraphBatches();
 */

class device::nodeBatched : public device::node {
protected:
	const size_t batch_size = 1;

public:
	void resizeBatch(const size_t new_batch_size) {
		if (batch_size != new_batch_size) {
			const_cast <size_t&> (batch_size) = new_batch_size;
			if (fwd != nullptr) {
				cudaGraphDestroy(fwd);
				fwd = nullptr;
			}
			if (back != nullptr) {
				cudaGraphDestroy(back);
				back = nullptr;
			}
		}
	}
};