#include "nodeBatched.cuh"

void device::nodeBatched::resizeBatch (const size_t new_batch_size) {
	if (batch_size != new_batch_size) {
		const_cast <size_t&>(batch_size) = new_batch_size;
		fwd.invalidate();
		back.invalidate();
	}
}