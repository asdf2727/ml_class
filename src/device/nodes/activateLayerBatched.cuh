#pragma once

#include "../../link.cuh"
#include "abstract/nodeBatched.cuh"
#include "../neurons/neuronArrayBatched.cuh"


namespace device {
	struct activationFunction;
	class activateLayerBatched;
}

struct device::activationFunction {
	float (*fwdFunc) (const float val);
	float (*backFunc) (const float act);

	activationFunction (float (*fwdFunc) (const float val), float (*backFunc) (const float act)) :
		fwdFunc(fwdFunc),
		backFunc(backFunc) {}

	bool operator== (const activationFunction &other) const {
		return fwdFunc == other.fwdFunc && backFunc == other.backFunc;
	}
};

class device::activateLayerBatched : public device::nodeBatched {
	link <device::neuronArrayBatched> data;

	void buildForward (device::graph *&fwd) override;
	void buildBackward (device::graph *&back) override;

public:
	const activationFunction act;

	explicit activateLayerBatched (device::neuronArrayBatched &data,
	                               const activationFunction &act) :
		data(data),
		act(act) {}

	void changeData (device::neuronArrayBatched &new_data);
};
