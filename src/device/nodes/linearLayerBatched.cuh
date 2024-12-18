#pragma once

#include "abstract/weightedNode.cuh"
#include "../device.cuh"
#include "../mem/matrix.cuh"
#include "../mem/neuronArrayBatched.cuh"
#include "abstract/IONodeBatched.cuh"

namespace device {
	class linearLayerBatched;
};

class device::linearLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
	device::neuronArrayBatched *input;
	device::neuronArrayBatched *output;

public:
	const size_t in_size, out_size, batch_size;

private:
	static constexpr float const1 = 1.0, const0 = 0.0;
	cublasHandle_t handle;

	device::matrix <float> mul;
	device::matrix <float> *mul_der = nullptr;
	device::matrix <float> *mul_der_sum = nullptr;

	void initMulDer() {
		if (mul_der == nullptr) mul_der = new device::matrix <float>(in_size + 1, out_size);
		if (mul_der_sum == nullptr) {
			mul_der_sum = new device::matrix <float>(in_size + 1, out_size);
			mul_der_sum->set(0x00);
		}
	}

	inline void makeForwardGraph() override;
	inline void makeBackwardGraph() override;

public:
	linearLayerBatched (device::neuronArrayBatched &input, device::neuronArrayBatched &output) :
	input(&input), output(&output), in_size(input.size), out_size(output.size), batch_size(input.batch_size), mul(in_size, out_size) {
		assert(input.batch_size == output.batch_size);
		cublasCreate(&handle);
	}

	~linearLayerBatched() {
		cublasDestroy(handle);
		delete input;
		delete output;
		delete mul;
		delete mul_der;
		delete mul_der_sum;
	}

	bool editInput(device::neuronArrayBatched &input) {
		if (this->input != &input) {
			this->input = &input;
			invalidateGraphs();
			return true;
		}
		return false;
	}
	bool editOutput(device::neuronArrayBatched &output) {
		if (this->output != &output) {
			this->output = &output;
			invalidateGraphs();
			return true;
		}
		return false;
	}

	void resetWeights(const float mean, const float std_dev, const unsigned long long seed = 0) {
		curandGenerator_t eng;
		curandCreateGenerator(&eng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(eng, seed);
		curandGenerateNormal(eng, mul, batch_size * mul.pitch, mean, std_dev);
		curandDestroyGenerator(eng);
		if (mul_der_sum != nullptr) mul_der_sum->set(0x00);
	}

	void loadWeights(const std::vector <float> &weights) {
		assert(weights.size() == (in_size + 1) * out_size);
		mul = weights.data();
		if (mul_der_sum != nullptr) mul_der_sum->set(0x00);
	}
	std::vector <float> saveWeights() const {
		std::vector <float> ans((in_size + 1) * out_size);
		mul.toHost(ans.data());
		return ans;
	}

	void descend(const float step_size, const cudaStream_t stream);
};

inline void device::linearLayerBatched::descend(const float step_size, const cudaStream_t stream) {
	initMulDer();
	cublasSetStream(handle, stream);
	// mult[(in+1)*out] = mult[(in+1)*out] + step_size * mult_der_sum[(in+1)*out]
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_size + 1, out_size,
		&const1, mul, mul.pitch,
		&step_size, *mul_der_sum, mul_der_sum->pitch,
		mul, mul.pitch);
	mul_der_sum->set(0x00);
}

inline void device::linearLayerBatched::makeForwardGraph() {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cublasSetStream(virtual_handle, stream);
	// output[out*batch] = trans(mult)[out*(in+1)] * input[(in+1)*batch]
	cublasSgemm(virtual_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_size, batch_size, in_size + 1,
		&const1, mul, mul.pitch, input.val, input.val.pitch,
		&const0, output.val, output.val.pitch);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream, &fwd);
	cudaStreamDestroy(stream);
}

inline void device::linearLayerBatched::makeBackwardGraph() {
	initMulDer();
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
	// mult_der[(in+1)*out] = input[(in+1)*batch] * trans(output_der)[batch*out]
	cublasSgemm(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_T, in_size + 1, out_size, batch_size,
		&const1, input.val, input.val.pitch, output.der, output.der.pitch,
		&const0, *mul_der, mul_der->pitch);
	// mult_der_sum[(in+1)*out] = mult_der_sum[(in+1)*out] + mult_der[(in+1)*out]
	cublasSgeam(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_size + 1, out_size,
		&const1, *mul_der_sum, mul_der_sum->pitch,
		&const1, *mul_der, mul_der->pitch,
		*mul_der_sum, mul_der_sum->pitch);

	cublasSetStream(virtual_handle, stream2);
	// input_der[in*batch] = mult[in*out] * output_der[out*batch]
	cublasSgemm(virtual_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_size, batch_size, out_size,
		&const1, mul, mul.pitch, output.der, output.der.pitch,
		&const0, input.der, input.der.pitch);

	cudaEvent_t event2 = nullptr;
	cudaEventCreate(&event2);
	cudaEventRecord(event2, stream2);
	cudaStreamWaitEvent(stream1, event2);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream1, &back);
	cudaEventDestroy(event1);
	cudaEventDestroy(event2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}