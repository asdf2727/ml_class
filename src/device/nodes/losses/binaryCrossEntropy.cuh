#pragma once

#include "../calculateLossBatched.cuh"

__device__ inline float BCEFunc(const float output, const float expected) {
	return -(expected * log(output) + (1 - expected) * log(1 - output));
}
__device__ inline float BCEDer(const float output, const float expected) {
	return (output - expected) / (output * (1 - output));
}
namespace device {
	const lossFunction BCE(BCEFunc, BCEDer);
}