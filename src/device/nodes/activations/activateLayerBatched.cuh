#pragma once

#include "../abstract/nodeBatched.cuh"
#include "../../mem/neuronArrayBatched.cuh"

namespace device {
	template <float (*fwdFunc)(float val), float (*backFunc)(float act)>
	class activateLayerBatched;
};

template <float (*fwdFunc)(float val), float (*backFunc)(float act)>
class device::activateLayerBatched : public device::nodeBatched {
	device::neuronArrayBatched &data;

	void makeForwardGraph ();
	inline void makeBackwardGraph();

public:
	explicit activateLayerBatched(device::neuronArrayBatched &data) : data(data) { }
};

template <float (*fwdFunc)(float val)>
__global__ void devActivateMatrix(float *mat, const size_t pitch, const size_t width, const size_t height) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		mat[idx + pitch * idy] = fwdFunc(mat[idx + pitch * idy]);
	}
}

template <float(*fwdFunc) (float val), float(*backFunc) (float act)>
void device::activateLayerBatched <fwdFunc, backFunc>::makeForwardGraph () {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(, blockDim.x), calcBlocks(data.batch_size, blockDim.y));
	devActivateMatrix <fwdFunc><<<gridDim, blockDim, 0, stream>>>
	(data.val, data.val.pitch, data.size, data.batch_size);

	cudaStreamEndCapture(stream, &fwd);
	cudaStreamDestroy(stream);
}

template <float (*backFunc)(float act)>
__global__ void devActivateMatrixDer(float *val, const size_t val_pitch, float *der, const size_t der_pitch, const size_t width, const size_t height) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		der[idx + der_pitch * idy] *= backFunc(val[idx + val_pitch * idy]);
	}
}

template <float(*fwdFunc) (float val), float(*backFunc) (float act)>
void device::activateLayerBatched <fwdFunc, backFunc>::makeBackwardGraph () { {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(data.size, blockDim.x), calcBlocks(data.batch_size, blockDim.y));
	devActivateMatrixDer <backFunc><<<gridDim, blockDim, 0, stream>>>
	(data.val, data.val.pitch, data.der, data.der.pitch, data.size, data.batch_size);

	cudaStreamEndCapture(stream, &back);
	cudaStreamDestroy(stream);
}
}