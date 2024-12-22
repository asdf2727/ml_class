#pragma once

#include "../activateLayerBatched.cuh"

__device__ inline float ReLUFunc(const float value) {
	return value > 0 ? value : 0;
}
__device__ inline float ReLUDer(const float value) {
	return value > 0 ? 1 : 0;
}
namespace device {
	const activationFunction ReLU(ReLUFunc, ReLUDer);
}