#include "common.cuh"

device::activationFunction device::getActFunc(const device::activationType type) {
	switch (type) {
		case device::ACT_NONE:
			return device::linear;
		case device::ACT_SIGMOID:
			return device::sigmoid;
		case device::ACT_RELU:
			return device::ReLU;
		default:
			throw std::invalid_argument("Unknown activation type " + std::to_string(type));
	}
}
device::activationType device::getActType(const device::activationFunction func) {
	if (func == device::linear) {
		return device::ACT_NONE;
	}
	if (func == device::sigmoid) {
		return device::ACT_SIGMOID;
	}
	if (func == device::ReLU) {
		return device::ACT_RELU;
	}
	return device::ACT_UNKNOWN;
}