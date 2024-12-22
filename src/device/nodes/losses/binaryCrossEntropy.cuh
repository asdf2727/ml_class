#pragma once

#include "../calculateLossBatched.cuh"

__device__ float BCEFunc(const float output, const float expected);
__device__ float BCEDer(const float output, const float expected);

namespace device {
	const lossFunction BCE(BCEFunc, BCEDer);
}