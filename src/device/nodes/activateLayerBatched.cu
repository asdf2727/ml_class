#include "activateLayerBatched.cuh"

__global__ void devActivateMatrix (float *mat, const size_t pitch, const size_t width,
                                   const size_t height, float (*fwdFunc) (float val)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) { mat[idx + pitch * idy] = fwdFunc(mat[idx + pitch * idy]); }
}

void device::activateLayerBatched::buildForward (device::graph *&fwd) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(data->getSize(), blockDim.x), calcBlocks(getBatchSize(), blockDim.y));
	devActivateMatrix<<<gridDim, blockDim, 0, stream>>>(data->val, data->val.pitch, data->getSize(),
	                                                    getBatchSize(), act.fwdFunc);

	cudaStreamEndCapture(stream, (cudaGraph_t*)fwd);
	cudaStreamDestroy(stream);
}

__global__ void devActivateMatrixDer (const size_t width, const size_t height,
                                      const float *val, const size_t val_pitch,
                                      float *der, const size_t der_pitch,
                                      float (*backFunc) (float act)) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < width && idy < height) {
		der[idx + der_pitch * idy] *= backFunc(val[idx + val_pitch * idy]);
	}
}

void device::activateLayerBatched::buildBackward (device::graph *&back) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(data->getSize(), blockDim.x), calcBlocks(getBatchSize(), blockDim.y));
	devActivateMatrixDer<<<gridDim, blockDim, 0, stream>>>(data->getSize(), getBatchSize(),
	                                                    data->val, data->val.pitch,
	                                                    *data->der, data->der->pitch,
	                                                       act.backFunc);

	cudaStreamEndCapture(stream, (cudaGraph_t*)back);
	cudaStreamDestroy(stream);
}

void device::activateLayerBatched::changeData (device::neuronArrayBatched &new_data) {
	assert(data->getSize() == new_data.getBatchSize() && data->getSize() == new_data.getBatchSize());
	data = new_data;
	fwd.invalidate();
	back.invalidate();
}
