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
	const size_t max_batch_size;
	const size_t batch_size;

public:
	explicit nodeBatched (const size_t max_batch_size) :
		max_batch_size(max_batch_size),
		batch_size(max_batch_size) {}

	nodeBatched () : max_batch_size(0), batch_size(0) {}

	nodeBatched (const nodeBatched &other) = delete;
	nodeBatched &operator= (const nodeBatched &other) = delete;
	nodeBatched (nodeBatched &&other) noexcept :
		max_batch_size(other.max_batch_size),
		batch_size(other.batch_size) {}
	nodeBatched &operator= (nodeBatched &&other) noexcept {
		if (this == &other) return *this;
		(size_t &)max_batch_size = other.max_batch_size;
		(size_t &)batch_size = other.batch_size;
		return *this;
	}
	~nodeBatched() override = default;

	void resizeBatch (const size_t new_batch_size);

	size_t getBatchSize() const { return batch_size; }
};
