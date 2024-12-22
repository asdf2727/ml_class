#include "common.cuh"

device::lossFunction device::getLossFunc (const device::lossType type) {
	switch (type) {
		case device::LOSS_MSE:
			return device::MSE;
		case device::LOSS_BCE:
			return device::BCE;
		default:
			throw std::invalid_argument("Unknown loss type " + std::to_string(type));
	}
}
device::lossType device::getLossType (const device::lossFunction func) {
	if (func == device::MSE) { return device::LOSS_MSE; }
	if (func == device::BCE) { return device::LOSS_BCE; }
	return device::LOSS_UNKNOWN;
}
