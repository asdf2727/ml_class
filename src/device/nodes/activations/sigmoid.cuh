#pragma once

#include "../activateLayerBatched.cuh"

__device__ inline float sigmoidFunc(const float value) {
	return 1 / (1 + expf(-value));
}
__device__ inline float sigmoidDer(const float sigmoid) {
	return sigmoid * (1 - sigmoid);
}
namespace device {
	constexpr activationFunction sigmoid(sigmoidFunc, sigmoidDer);
};