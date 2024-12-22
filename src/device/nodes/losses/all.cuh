#pragma once

#include "meanSquaredError.cuh"
#include "binaryCrossEntropy.cuh"

namespace device {
	enum lossType {
		LOSS_UNKNOWN = 0,
		LOSS_MSE = 1,
		LOSS_BCE = 2
	};
	device::lossFunction getLossFunc(const device::lossType type);
	device::lossType getLossType(const device::lossFunction func);
}

device::lossFunction getLossFunc (const device::lossType type) {
	switch (type) {
		case device::LOSS_MSE:
			return device::MSE;
		case device::LOSS_BCE:
			return device::BCE;
		default:
			throw std::invalid_argument("Unknown loss type " + std::to_string(type));
	}
}
device::lossType getLossType (const device::lossFunction func) {
	if (func == device::MSE) {
		return device::LOSS_MSE;
	}
	if (func == device::BCE) {
		return device::LOSS_BCE;
	}
	return device::LOSS_UNKNOWN;
}
