#include "linearLayerBatched.cuh"

#include <curand.h>

void device::linearLayerBatched::buildForward (device::graph *&fwd) {
	cudaGraphCreate((cudaGraph_t*)fwd, 0);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cublasSetStream(virtual_handle, stream);
	// output[out*batch] = trans(mult)[out*(in+1)] * input[(in+1)*batch]
	cublasSgemm(virtual_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_size, batch_size, in_size + 1,
	            &const1, mul, mul.pitch, input->val, input->val.pitch,
	            &const0, output->val, output->val.pitch);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream, (cudaGraph_t*)fwd);
	cudaStreamDestroy(stream);
}

void device::linearLayerBatched::buildBackward (device::graph *&back) {
	cudaGraphCreate((cudaGraph_t*)back, 0);
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamBeginCapture(stream1, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cudaEvent_t event1 = nullptr;
	cudaEventCreate(&event1);
	cudaEventRecord(event1, stream1);
	cudaStreamWaitEvent(stream2, event1);

	cublasSetStream(virtual_handle, stream1);
	// mult_der[(in+1)*out] += input[(in+1)*batch] * trans(output_der)[batch*out]
	cublasSgemm(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_T, in_size + 1, out_size, batch_size,
	            &const1, input->val, input->val.pitch, *output->der, output->der->pitch,
	            &const1, *mul_der, mul_der->pitch);

	cublasSetStream(virtual_handle, stream2);
	// input_der[in*batch] = mult[in*out] * output_der[out*batch]
	cublasSgemm(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_size, batch_size, out_size, &const1,
	            mul, mul.pitch, *output->der, output->der->pitch,
	            &const0, *input->der, input->der->pitch);

	cudaEvent_t event2 = nullptr;
	cudaEventCreate(&event2);
	cudaEventRecord(event2, stream2);
	cudaStreamWaitEvent(stream1, event2);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream1, (cudaGraph_t*)back);
	cudaEventDestroy(event1);
	cudaEventDestroy(event2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

void device::linearLayerBatched::buildDescent (device::graph *&desc) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cublasSetStream(virtual_handle, stream);
	// mult[(in+1)*out] = mult[(in+1)*out] + step_size * mult_der_sum[(in+1)*out]
	cublasSgeam(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_size + 1, out_size,
	            &const1, mul, mul.pitch,
	            &step_size, *mul_der, mul_der->pitch, mul, mul.pitch);
	mul_der->set(0x00, stream);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream, (cudaGraph_t*)desc);
	cudaStreamDestroy(stream);
}

void device::linearLayerBatched::resetWeights (const float mean, const float std_dev, const unsigned long long seed)  {
	curandGenerator_t eng;
	curandCreateGenerator(&eng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(eng, seed);
	curandGenerateNormal(eng, mul, batch_size * mul.pitch, mean, std_dev);
	curandDestroyGenerator(eng);
	mul_der->set(0x00);
}

void device::linearLayerBatched::loadWeights (const std::vector <float> &weights) {
	assert(weights.size() == (in_size + 1) * out_size);
	mul = weights.data();
	mul_der->set(0x00);
}

inline std::vector <float> device::linearLayerBatched::saveWeights () const {
	std::vector <float> ans((in_size + 1) * out_size);
	mul.toHost(ans.data());
	return ans;
}
