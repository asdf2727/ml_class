#include "calculateLossBatched.cuh"

__global__ void devCalculateLoss (const size_t size, const size_t batch_size, float *output,
								  const size_t output_pitch, float *output_der,
								  const size_t output_der_pitch, const float *expected,
								  const size_t expected_pitch, const device::lossFunction loss) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < size && idy < batch_size) {
		output_der[idx + output_der_pitch * idy] =
				loss.der(output[idx + output_pitch * idy], expected[idx + expected_pitch * idy]);
		output[idx + output_pitch * idy] =
				loss.val(output[idx + output_pitch * idy], expected[idx + expected_pitch * idy]);
	}
}

void device::calculateLossBatched::buildGraph (device::graph *&graph) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(output->getSize(), blockDim.x), calcBlocks(output->getBatchSize(), blockDim.y));
	devCalculateLoss<<<gridDim, blockDim, 0, stream>>>(output->getSize(), output->getBatchSize(),
													   output->val, output->val.pitch,
													   *output->der, output->der->pitch,
													   *expected, expected->pitch, loss);

	cudaStreamEndCapture(stream, (cudaGraph_t*)graph);
	cudaStreamDestroy(stream);
}

void device::calculateLossBatched::changeExpected (device::matrix <float> &new_expected) {
	assert(expected->X == new_expected.X && expected->Y == new_expected.Y);
	expected = new_expected;
	graph.invalidate();
}