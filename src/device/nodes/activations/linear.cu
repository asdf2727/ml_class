#include "linear.cuh"

__device__ float linearFunc(const float value) {
	return value > 0 ? value : 0;
}
__device__ float linearDer(const float value) {
	return value > 0 ? 1 : 0;
}