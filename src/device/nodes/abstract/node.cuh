#pragma once

#include "../../wrappers/graph.cuh"
#include "../../../lazy.cuh"

namespace device {
	class node;
}

/* Implement:
 * void makeForwardGraph();
 * void makeBackwardGraph();
 */

class device::node {
protected:
	virtual void buildForward (device::graph *&fwd) = 0;
	lazy <device::graph> fwd = lazy <device::graph>
			([this](device::graph *&fwd) { buildForward(fwd); });

	virtual void buildBackward (device::graph *&back) = 0;
	lazy <device::graph> back = lazy <device::graph>
			([this](device::graph *&back) { buildBackward(back); });

public:
	virtual ~node () = default;

	device::graph &getForward () { return *fwd; }
	device::graph &getBackward () { return *back; }
};
