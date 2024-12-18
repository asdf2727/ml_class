#pragma once

#include "abstract/weightedNodeBatched.cuh"
#include "abstract/nodeBatched.cuh"
#include "../mem/neuronArrayBatched.cuh"

__device__ inline float linearFunc(const float value) {
	return value;
}
__device__ inline float linearDer(const float value) {
	return 1;
}

namespace device {
	struct activationFunction {
		float (*fwdFunc)(const float val);
		float (*backFunc)(const float act);

		activationFunction(float (*fwdFunc)(const float val), float (*backFunc)(const float act)) : fwdFunc(fwdFunc), backFunc(backFunc) { }

		bool operator== (const activationFunction &other) const {
			return fwdFunc == other.fwdFunc && backFunc == other.backFunc;
		}
	};

	constexpr activationFunction linear(linearFunc, linearDer);

	class activateLayerBatched;
};

class device::activateLayerBatched : public device::nodeBatched {
	device::neuronArrayBatched *data;

	const activationFunction act;

	void makeForwardGraph () override;
	void makeBackwardGraph () override;

public:
	explicit activateLayerBatched(device::neuronArrayBatched &data, const activationFunction act) :
	data(&data), act(act) { }

	bool editData (device::neuronArrayBatched &data) {
		if (this->data != &data) {
			this->data = &data;
			invalidateGraphs();
			return true;
		}
		return false;
	}
};

__global__ void devActivateMatrix(float *mat, const size_t pitch, const size_t width, const size_t height, float (*fwdFunc)(float val)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		mat[idx + pitch * idy] = fwdFunc(mat[idx + pitch * idy]);
	}
}

void device::activateLayerBatched::makeForwardGraph () {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(, blockDim.x), calcBlocks(batch_size, blockDim.y));
	devActivateMatrix<<<gridDim, blockDim, 0, stream>>>
	(data.val, data.val.pitch, data.size, batch_size, act.fwdFunc);

	cudaStreamEndCapture(stream, &fwd);
	cudaStreamDestroy(stream);
}

__global__ void devActivateMatrixDer(float *val, const size_t val_pitch, float *der, const size_t der_pitch, const size_t width, const size_t height, float (*backFunc)(float act)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		der[idx + der_pitch * idy] *= backFunc(val[idx + val_pitch * idy]);
	}
}

void device::activateLayerBatched::makeBackwardGraph () {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(data.size, blockDim.x), calcBlocks(batch_size, blockDim.y));
	devActivateMatrixDer<<<gridDim, blockDim, 0, stream>>>
	(data.val, data.val.pitch, data.der, data.der.pitch, data.size, batch_size, act.backFunc);

	cudaStreamEndCapture(stream, &back);
	cudaStreamDestroy(stream);
}