#pragma once

#include <vector>
#include <queue>

#include "../mem/neuronArrayBatched.cuh"
#include "../nodes/activations/sigmoid.cuh"
#include "../nodes/denseLayerBatched.cuh"

namespace device {
	class multiLayerPerceptronBuilder;
	class multiLayerPerceptron;
};

class device::multiLayerPerceptron {
	std::vector <device::IONodeBatched <device::neuronArrayBatched> *> layers;

	cudaGraph_t build_fwd = nullptr, build_back = nullptr;
	bool updateForwardBuildGraph();
	bool updateBackwardBuildGraph();
	void invalidateBuildGraphs();

	cudaGraphExec_t fwd = nullptr, back = nullptr;
	void updateForwardGraphExec();
	void updateBackwardGraphExec();

	std::vector <device::neuronArrayBatched> arrays;
	device::matrix <float> exp_output;

	device::neuronArrayBatched in_buffer;
	device::matrix <float> out_buffer;

	void switchInBuffer();
	void switchOutBuffer();
	void switchTrainBuffer();

	const size_t max_batch_size;
	size_t exec_batch_size;
	size_t batch_cnt = 0;

	void updateBatchSize();

	cudaStream_t stream;
	bool train_mode = false;

	void run();
	void train();

	std::queue <std::vector <float>> out_queue;


public:
	multiLayerPerceptron(const std::vector <size_t> &sizes, const std::vector <device::activationFunction *> &activations, const size_t batch_size) {
		arrays.emplace_back(sizes[0], max_batch_size);
		for (size_t i = 1; i < sizes.size() - 1; i++) {
			arrays.emplace_back(sizes[i], max_batch_size);
			layers.push_back(activations[i] == nullptr ?
				new device::denseLayerBatched(arrays[i - 1], arrays[i], *activations[i]) :
				new device::linearLayerBatched(arrays[i - 1], arrays[i]));
		}

		cudaStreamCreate(&stream);
		exec_batch_size = max_batch_size;
	}

	~multiLayerPerceptron() {
		for (const device::weightedNodeBatched *layer : layers) {
			delete layer;
		}
	}

	void bufferRun(const std::vector <float> &input);
	void bufferTrain(const std::vector <float> &input, const std::vector <float> &exp_output);

	std::vector <float> getOutput() {
		if (out_queue.empty()) {
			run();
		}
		std::vector <float> output = std::move(out_queue.front());
		out_queue.pop();
		return output;
	}
};

inline bool device::multiLayerPerceptron::updateForwardBuildGraph () {
	if (build_fwd == nullptr) {
		return false;
	}
	cudaGraphCreate(&build_fwd, 0);
	cudaGraphNode_t last_node = nullptr, this_node;
	for (device::weightedNodeBatched *layer : layers) {
		cudaGraphAddChildGraphNode(&this_node, build_fwd, &last_node, 0, layer->getForwardGraph());
		last_node = this_node;
	}
	return true;
}
inline bool device::multiLayerPerceptron::updateBackwardBuildGraph () {
	if (build_back != nullptr) {
		return false;
	}
	cudaGraphCreate(&build_back, 0);
	cudaGraphNode_t last_node = nullptr, this_node;
	for (device::weightedNodeBatched *layer : layers) {
		cudaGraphAddChildGraphNode(&this_node, build_back, &last_node, 0, layer->getForwardGraph());
		last_node = this_node;
	}
	// TODO add error calculation
	for (size_t i = layers.size() - 1; i < layers.size(); i--) {
		cudaGraphAddChildGraphNode(&this_node, build_back, &last_node, 0, layers[i]->getBackwardGraph());
		last_node = this_node;
	}
	return true;
}
inline void device::multiLayerPerceptron::invalidateBuildGraphs() {
	if (build_fwd != nullptr) {
		cudaGraphDestroy(build_fwd);
		build_fwd = nullptr;
	}
	if (build_back != nullptr) {
		cudaGraphDestroy(build_back);
		build_back = nullptr;
	}
}

inline void device::multiLayerPerceptron::updateForwardGraphExec() {
	if (updateForwardBuildGraph()) {
		if (!train_mode) {
			cudaStreamSynchronize(stream);
		}
		cudaGraphExecForceUpdate(&fwd, build_fwd);
	}
}
inline void device::multiLayerPerceptron::updateBackwardGraphExec() {
	if (updateBackwardBuildGraph()) {
		if (train_mode) {
			cudaStreamSynchronize(stream);
		}
		cudaGraphExecForceUpdate(&back, build_back);
	}
}

inline void device::multiLayerPerceptron::switchInBuffer() {
	std::swap(in_buffer, arrays.front());
	layers[0]->editInput(arrays.front());
	invalidateBuildGraphs();
}
inline void device::multiLayerPerceptron::switchOutBuffer() {
	std::swap(out_buffer, arrays.back().val);
	layers[0]->editInput(arrays.back());
	invalidateBuildGraphs();
}
inline void device::multiLayerPerceptron::switchTrainBuffer() {
	std::swap(out_buffer, exp_output);
	// TODO add edit exp_out
	invalidateBuildGraphs();
}

inline void device::multiLayerPerceptron::updateBatchSize() {
	if (batch_cnt != exec_batch_size) {
		for (const auto layer : layers) {
			layer->resizeBatch(batch_cnt);
		}
		invalidateBuildGraphs();
		exec_batch_size = batch_cnt;
	}
}

inline void device::multiLayerPerceptron::run() {
	const size_t old_batch_size = exec_batch_size;
	if (batch_cnt == 0) {
		// skip enqueuing if empty buffer
		cudaStreamSynchronize(stream);
		exec_batch_size = 0;
	}
	else {
		// enqueue new run
		updateBatchSize();
		updateForwardGraphExec();
		cudaStreamSynchronize(stream);	// wait for the last action to finish
		cudaGraphLaunch(fwd, stream);	// then start this one
		switchInBuffer();
		switchOutBuffer();
		batch_cnt = 0;
	}
	train_mode = false;
	// get old results from output buffer
	for (size_t i = 0; i < old_batch_size; i++) {
		out_queue.emplace(out_buffer.X);
		cudaMemcpy(out_queue.back().data(), out_buffer + out_buffer.pitch * i,
			sizeof(float) * out_buffer.X, cudaMemcpyDeviceToHost);
	}
}
inline void device::multiLayerPerceptron::train() {
	if (batch_cnt == 0) {
		// skip enqueuing if empty buffer
		exec_batch_size = 0;
	}
	else {
		// enqueue new run
		updateBatchSize();
		updateBackwardGraphExec();
		cudaStreamSynchronize(stream);	// wait for the last action to finish
		cudaGraphLaunch(back, stream);	// then start this one
		// TODO add descent
		switchInBuffer();
		switchTrainBuffer();
		batch_cnt = 0;
	}
	train_mode = true;
}

inline void device::multiLayerPerceptron::bufferRun(const std::vector <float> &input) {
	if (batch_cnt == max_batch_size) {
		run();
	}
	cudaMemcpy(in_buffer.val + in_buffer.val.pitch * batch_cnt,
		input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
	batch_cnt++;
}
inline void device::multiLayerPerceptron::bufferTrain(const std::vector <float> &input, const std::vector <float> &exp_output) {
	if (batch_cnt == max_batch_size) {
		train();
	}
	cudaMemcpy(in_buffer.val + in_buffer.val.pitch * batch_cnt, input.data(),
		input.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(out_buffer + out_buffer.pitch * batch_cnt, exp_output.data(),
		exp_output.size() * sizeof(float), cudaMemcpyHostToDevice);
	batch_cnt++;
}