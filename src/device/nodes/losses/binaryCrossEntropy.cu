#include "binaryCrossEntropy.cuh"

__device__ float BCEFunc(const float output, const float expected) {
	return -(expected * log(output) + (1 - expected) * log(1 - output));
}
__device__ float BCEDer(const float output, const float expected) {
	return (output - expected) / (output * (1 - output));
}