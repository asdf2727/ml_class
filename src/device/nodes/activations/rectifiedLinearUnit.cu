#include "rectifiedLinearUnit.cuh"

__device__ float ReLUFunc(const float value) {
	return value > 0 ? value : 0;
}
__device__ float ReLUDer(const float value) {
	return value > 0 ? 1 : 0;
}