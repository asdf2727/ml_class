#pragma once

#include "../../wrappers/graph.cuh"
#include "../../../lazy.cuh"

namespace device {
	class node;
};

/* Implement:
 * void makeForwardGraph();
 * void makeBackwardGraph();
 */

class device::node {
protected:
	virtual void buildForward(const device::graph *&fwd) const;
	lazy <device::graph, device::node> fwd;

	virtual void buildBackward(const device::graph *&back) const;
	lazy <device::graph, device::node> back;

public:
	node() : fwd(buildForward), back(buildBackward) { }

	virtual ~node() = default;

	const device::graph &getForward() const { return fwd; }
	const device::graph &getBackward() const { return back; }
};