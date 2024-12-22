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

	activationFunction() : fwdFunc(nullptr), backFunc(nullptr) {}

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
		nodeBatched(data.getBatchSize()),
		data(data),
		act(act) {}

	activateLayerBatched (const activateLayerBatched &other) = delete;
	activateLayerBatched &operator= (const activateLayerBatched &other) = delete;
	activateLayerBatched (activateLayerBatched &&other) noexcept :
		nodeBatched(std::move(other)),
		data(std::move(other.data)),
		act(std::move(other.act)) {}
	activateLayerBatched &operator= (activateLayerBatched &&other) noexcept {
		if (this == &other) return *this;
		nodeBatched::operator=(std::move(other));
		data = std::move(other.data);
		(activationFunction &)act = std::move(other.act);
		return *this;
	}
	~activateLayerBatched () override = default;

	void changeData (device::neuronArrayBatched &new_data);
};
