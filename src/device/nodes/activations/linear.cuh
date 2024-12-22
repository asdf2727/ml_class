#pragma once

#include "../activateLayerBatched.cuh"

__device__ float linearFunc(const float value);
__device__ float linearDer(const float value);

namespace device {
	const activationFunction linear(linearFunc, linearDer);
}