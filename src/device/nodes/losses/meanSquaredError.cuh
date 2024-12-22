#pragma once

#include "../calculateLossBatched.cuh"

__device__ inline float MSEFunc(const float output, const float expected) {
	return (output - expected) * (output - expected);
}
__device__ inline float MSEDer(const float output, const float expected) {
	return 2 * (output - expected);
}

namespace device {
	const lossFunction MSE(MSEFunc, MSEDer);
}