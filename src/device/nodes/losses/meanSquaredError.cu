#include "meanSquaredError.cuh"

__device__ float MSEFunc(const float output, const float expected) {
	return (output - expected) * (output - expected);
}
__device__ float MSEDer(const float output, const float expected) {
	return 2 * (output - expected);
}