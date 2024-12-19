#pragma once

#include "../../link.cuh"
#include "../../lazy.cuh"
#include "../wrappers/graph.cuh"
#include "../wrappers/matrix.cuh"
#include "../neurons/neuronArrayBatched.cuh"

namespace device {
	struct lossFunction;
	class calculateLossBatched;
};

struct device::lossFunction {
	float (*loss) (const float output, const float expected);
	float (*lossDer) (const float output, const float expected);

	lossFunction (float (*loss) (const float output, const float expected), float (*lossDer) (const float output, const float expected)) :
		loss(loss),
		lossDer(lossDer) {}
};

class device::calculateLossBatched {
	link <device::neuronArrayBatched> output;
	link <device::matrix <float>> expected;

	void buildGraph (const device::graph *&graph) const;
	lazy <device::graph, device::calculateLossBatched> graph;

	const lossFunction &loss;

public:
	calculateLossBatched (device::neuronArrayBatched &output,
	                      device::matrix <float> &expected,
	                      const device::lossFunction &loss) : output(output),
	                                                          expected(expected),
	                                                          graph(buildGraph),
	                                                          loss(loss) {
		assert(output.size == expected.X && output.batch_size == expected.Y);
	}

	void changeOutput (device::neuronArrayBatched &new_output) {
		assert(output.get().size == new_output.size && output.get().batch_size == new_output.batch_size);
		output = new_output;
		graph.invalidate();
	}
	void changeExpected (device::matrix <float> &new_expected) {
		assert(expected.get().X == new_expected.X && expected.get().Y == new_expected.Y);
		expected = new_expected;
		graph.invalidate();
	}

	device::graph getGraph() const { return graph; }
};

__global__ inline void devCalculateLoss (const size_t size,
                                         const size_t batch_size,
                                         float *output,
                                         const size_t output_pitch,
                                         float *output_der,
                                         const size_t output_der_pitch,
                                         const float *expected,
                                         const size_t expected_pitch,
                                         const device::lossFunction &loss) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < size && idy < batch_size) {
		output_der[idx + output_der_pitch * idy] = loss.lossDer(output[idx + output_pitch * idy],
		                                                        expected[idx + expected_pitch * idy]);
		output[idx + output_pitch * idy] = loss.loss(output[idx + output_pitch * idy],
		                                             expected[idx + expected_pitch * idy]);
	}
}

void device::calculateLossBatched::buildGraph (const device::graph *&graph) const {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

	dim3 blockDim(16, 16);
	dim3 gridDim(calcBlocks(output.get().size, blockDim.x), calcBlocks(output.get().batch_size, blockDim.y));
	devCalculateLoss<<<gridDim, blockDim, 0, stream>>>(output.get().size,
	                                                   output.get().batch_size,
	                                                   output.get().val,
	                                                   output.get().val.pitch,
	                                                   output.get().der.get(),
	                                                   output.get().der.get().pitch,
	                                                   expected.get(),
	                                                   expected.get().pitch,
	                                                   loss);

	cudaStreamEndCapture(stream, (cudaGraph_t *)graph);
	cudaStreamDestroy(stream);
}
