#include "sigmoid.cuh"

__device__ float sigmoidFunc(const float value) {
	return 1 / (1 + expf(-value));
}
__device__ float sigmoidDer(const float sigmoid) {
	return sigmoid * (1 - sigmoid);
}
