#pragma once

#include "abstract/weightedNodeBatched.cuh"
#include "abstract/nodeBatched.cuh"
#include "../neurons/neuronArrayBatched.cuh"


namespace device {
	struct activationFunction;
	class activateLayerBatched;
};

struct device::activationFunction {
	float (*fwdFunc) (const float val);
	float (*backFunc) (const float act);

	activationFunction (float (*fwdFunc) (const float val), float (*backFunc) (const float act)) : fwdFunc(fwdFunc), backFunc(backFunc) {}

	bool operator== (const activationFunction &other) const {
		return fwdFunc == other.fwdFunc && backFunc == other.backFunc;
	}
};

__device__ inline float linearFunc (const float value) {
	return value;
}

__device__ inline float linearDer (const float value) {
	return 1;
}

constexpr device::activationFunction linear(linearFunc, linearDer);

class device::activateLayerBatched : public device::nodeBatched {
	link <device::neuronArrayBatched> data;

	const activationFunction act;

	void buildForward (const device::graph *&fwd) const override;
	void buildBackward (const device::graph *&back) const override;

public:
	explicit activateLayerBatched (device::neuronArrayBatched &data, const activationFunction &act) :
		data(data),
		act(act) {}

	void changeData (device::neuronArrayBatched &new_data) {
		assert(data.get().size == new_data.size && data.get().batch_size == new_data.batch_size);
		data = new_data;
		fwd.invalidate();
		back.invalidate();
	}
};

__global__ void devActivateMatrix (float *mat,
                                   const size_t pitch,
                                   const size_t width,
                                   const size_t height,
                                   float (*fwdFunc) (float val)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		mat[idx + pitch * idy] = fwdFunc(mat[idx + pitch * idy]);
	}
}

void device::activateLayerBatched::buildForward (const device::graph *&fwd) const {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(, blockDim.x), calcBlocks(batch_size, blockDim.y));
	devActivateMatrix<<<gridDim, blockDim, 0, stream>>>(data.get().val,
	                                                    data.get().val.pitch,
	                                                    data.get().size,
	                                                    batch_size,
	                                                    act.fwdFunc);

	cudaStreamEndCapture(stream, (cudaGraph_t *)fwd);
	cudaStreamDestroy(stream);
}

__global__ void devActivateMatrixDer (const float *val,
                                      const size_t val_pitch,
                                      float *der,
                                      const size_t der_pitch,
                                      const size_t width,
                                      const size_t height,
                                      float (*backFunc) (float act)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		der[idx + der_pitch * idy] *= backFunc(val[idx + val_pitch * idy]);
	}
}

void device::activateLayerBatched::buildBackward (const device::graph *&back) const {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(data.get().size, blockDim.x), calcBlocks(batch_size, blockDim.y));
	devActivateMatrixDer<<<gridDim, blockDim, 0, stream>>>(data.get().val,
	                                                       data.get().val.pitch,
	                                                       data.get().der.get(),
	                                                       data.get().der.get().pitch,
	                                                       data.get().size,
	                                                       batch_size,
	                                                       act.backFunc);

	cudaStreamEndCapture(stream, (cudaGraph_t *)back);
	cudaStreamDestroy(stream);
}
