#pragma once

#include "../activateLayerBatched.cuh"

__device__ float ReLUFunc(const float value);
__device__ float ReLUDer(const float value);

namespace device {
	const activationFunction ReLU(ReLUFunc, ReLUDer);
}