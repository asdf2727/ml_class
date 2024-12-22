#pragma once

#include "meanSquaredError.cuh"
#include "binaryCrossEntropy.cuh"

namespace device {
	enum lossType {
		LOSS_UNKNOWN = 0,
		LOSS_MSE     = 1,
		LOSS_BCE     = 2
	};

	device::lossFunction getLossFunc (const device::lossType type);
	device::lossType getLossType (const device::lossFunction func);
}