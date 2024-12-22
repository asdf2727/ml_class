#pragma once

#include "linear.cuh"
#include "sigmoid.cuh"
#include "rectifiedLinearUnit.cuh"

namespace device {
	enum activationType {
		ACT_UNKNOWN = 0,
		ACT_NONE    = 1,
		ACT_SIGMOID = 2,
		ACT_RELU    = 3,
	};
	device::activationFunction getActFunc(const device::activationType type);
	device::activationType getActType(const device::activationFunction func);
}