#pragma once

#include "abstract/weightedNode.cuh"
#include "../device.cuh"
#include "../wrappers/matrix.cuh"
#include "../neurons/neuronArrayBatched.cuh"
#include "abstract/IONodeBatched.cuh"

namespace device {
	class linearLayerBatched;
};

class device::linearLayerBatched : public device::IONodeBatched <device::neuronArrayBatched> {
public:
	const size_t &in_size, out_size, batch_size;

private:
	static constexpr float const1 = 1.0, const0 = 0.0;

	device::matrix <float> mul;
	void buildMultDer (const device::matrix <float> *mul_der) const {
		mul_der = new device::matrix <float>(in_size + 1, out_size);
		mul_der->set(0x00);
	}
	lazy <device::matrix <float>, device::linearLayerBatched> mul_der;
	float step_size = 0;


	void buildForward (const device::graph *&fwd) const override;
	void buildBackward (const device::graph *&back) const override;
	void buildDescent (const device::graph *&desc) const override;

public:
	linearLayerBatched (device::neuronArrayBatched &input,
	                    device::neuronArrayBatched &output) : IONodeBatched(input, output),
	                                                          in_size(this->input.get().size),
	                                                          out_size(this->output.get().size),
	                                                          batch_size(this->input.get().batch_size),
	                                                          mul(in_size, out_size),
	                                                          mul_der(buildMultDer) { }

	void resetWeights (const float mean, const float std_dev, const unsigned long long seed = 0) override {
		curandGenerator_t eng;
		curandCreateGenerator(&eng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(eng, seed);
		curandGenerateNormal(eng, mul, batch_size * mul.pitch, mean, std_dev);
		curandDestroyGenerator(eng);
		mul_der.get().set(0x00);
	}
	void loadWeights (const std::vector <float> &weights) override {
		assert(weights.size() == (in_size + 1) * out_size);
		mul = weights.data();
		mul_der.get().set(0x00);
	}

	std::vector <float> saveWeights () const override {
		std::vector <float> ans((in_size + 1) * out_size);
		mul.toHost(ans.data());
		return ans;
	}
};

void device::linearLayerBatched::buildForward (const device::graph *&fwd) const {
	cudaGraphCreate((cudaGraph_t *)fwd, 0);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cublasSetStream(virtual_handle, stream);
	// output[out*batch] = trans(mult)[out*(in+1)] * input[(in+1)*batch]
	cublasSgemm(virtual_handle,
	            CUBLAS_OP_T,
	            CUBLAS_OP_N,
	            out_size,
	            batch_size,
	            in_size + 1,
	            &const1,
	            mul,
	            mul.pitch,
	            input.get().val,
	            input.get().val.pitch,
	            &const0,
	            output.get().val,
	            output.get().val.pitch);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream, (cudaGraph_t *)fwd);
	cudaStreamDestroy(stream);
}

void device::linearLayerBatched::buildBackward (const device::graph *&back) const {
	cudaGraphCreate((cudaGraph_t *)fwd, 0);
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
	cublasSgemm(virtual_handle,
	            CUBLAS_OP_N,
	            CUBLAS_OP_T,
	            in_size + 1,
	            out_size,
	            batch_size,
	            &const1,
	            input.get().val,
	            input.get().val.pitch,
	            output.get().der.get(),
	            output.get().der.get().pitch,
	            &const1,
	            mul_der.get(),
	            mul_der.get().pitch);

	cublasSetStream(virtual_handle, stream2);
	// input_der[in*batch] = mult[in*out] * output_der[out*batch]
	cublasSgemm(virtual_handle,
	            CUBLAS_OP_N,
	            CUBLAS_OP_N,
	            in_size,
	            batch_size,
	            out_size,
	            &const1,
	            mul,
	            mul.pitch,
	            output.get().der.get(),
	            output.get().der.get().pitch,
	            &const0,
	            input.get().der.get(),
	            input.get().der.get().pitch);

	cudaEvent_t event2 = nullptr;
	cudaEventCreate(&event2);
	cudaEventRecord(event2, stream2);
	cudaStreamWaitEvent(stream1, event2);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream1, (cudaGraph_t *)back);
	cudaEventDestroy(event1);
	cudaEventDestroy(event2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

void device::linearLayerBatched::buildDescent (const device::graph *&desc) const {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	cublasHandle_t virtual_handle;
	cublasCreate(&virtual_handle);

	cublasSetStream(virtual_handle, stream);
	// mult[(in+1)*out] = mult[(in+1)*out] + step_size * mult_der_sum[(in+1)*out]
	cublasSgeam(virtual_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				in_size + 1,
				out_size,
				&const1,
				mul,
				mul.pitch,
				&step_size,
				mul_der.get(),
				mul_der.get().pitch,
				mul,
				mul.pitch);
	mul_der.get().set(0x00, stream);

	cublasDestroy(virtual_handle);
	cudaStreamEndCapture(stream, (cudaGraph_t *)desc);
	cudaStreamDestroy(stream);
}
