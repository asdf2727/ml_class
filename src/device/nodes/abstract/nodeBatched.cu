#include "nodeBatched.cuh"

void device::nodeBatched::resizeBatch (const size_t new_batch_size) {
	assert(new_batch_size <= max_batch_size);
	if (batch_size != new_batch_size) {
		(size_t &)batch_size = new_batch_size;
		fwd.invalidate();
		back.invalidate();
	}
}