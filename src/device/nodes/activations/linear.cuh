#pragma once

#include "../activateLayerBatched.cuh"

__device__ inline float linearFunc(const float value) {
	return value > 0 ? value : 0;
}
__device__ inline float linearDer(const float value) {
	return value > 0 ? 1 : 0;
}
namespace device {
	const activationFunction linear(linearFunc, linearDer);
}