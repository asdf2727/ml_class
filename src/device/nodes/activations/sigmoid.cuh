#pragma once

#include "../activateLayerBatched.cuh"

__device__ float sigmoidFunc(const float value);
__device__ float sigmoidDer(const float sigmoid);

namespace device {
	const activationFunction sigmoid(sigmoidFunc, sigmoidDer);
}