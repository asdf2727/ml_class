#pragma once

#include "node.cuh"

namespace device {
	class nodeBatched;
}

/* Implement:
 * void makeForwardGraph();
 * void makeForwardGraph();
 */
class device::nodeBatched : public virtual device::node {
public:
	const size_t batch_size = 1;

	void resizeBatch (const size_t new_batch_size);
};
