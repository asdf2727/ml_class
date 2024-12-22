#pragma once

#include "../calculateLossBatched.cuh"

__device__ float MSEFunc(const float output, const float expected);
__device__ float MSEDer(const float output, const float expected);

namespace device {
	const lossFunction MSE(MSEFunc, MSEDer);
}