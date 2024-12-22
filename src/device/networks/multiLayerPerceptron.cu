#include "multiLayerPerceptron.cuh"

#include "../../binaryIO.cuh"
#include "../nodes/activations/common.cuh"
#include "../nodes/losses/common.cuh"

// Private functions

cudaGraphNode_t putInGraph (device::graph &graph, const device::graph &subgraph) {
	cudaGraphNode_t node;
	cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, subgraph);
	return node;
}
void appendToGraph (cudaGraphNode_t &last, device::graph &graph,
                           const device::graph &subgraph) {
	cudaGraphNode_t node;
	cudaGraphAddChildGraphNode(&node, graph, &last, last == nullptr ? 1 : 0, subgraph);
	last = node;
}

void device::multiLayerPerceptron::buildForwardGraph (device::graph *&fwd) {
	cudaGraphCreate((cudaGraph_t*)fwd, 0);
	cudaGraphNode_t last_node = nullptr;
	for (device::denseLayerBatched &layer : layers) {
		appendToGraph(last_node, *fwd, layer.getForward());
	}
}
void device::multiLayerPerceptron::buildBackwardGraph (device::graph *&back) {
	cudaGraphCreate((cudaGraph_t*)back, 0);
	cudaGraphNode_t last_node = nullptr;
	for (device::denseLayerBatched &layer : layers) {
		appendToGraph(last_node, *back, layer.getForward());
	}
	appendToGraph(last_node, *back, loss.getGraph());
	for (size_t i = layers.size() - 1; i < layers.size(); i--) {
		appendToGraph(last_node, *back, layers[i].getBackward());
	}
}
void device::multiLayerPerceptron::buildDescentGraph (device::graph *&desc) {
	cudaGraphCreate((cudaGraph_t*)desc, 0);
	for (device::denseLayerBatched &layer : layers) { putInGraph(*back, layer.getDescent()); }
}

void device::multiLayerPerceptron::switchInBuffer () {
	std::swap(in_buffer, arrays.front());
	layers[0].changeInput(arrays.front());
	fwd.invalidate();
	back.invalidate();
	fwd_exec.invalidate();
	back_exec.invalidate();
}
void device::multiLayerPerceptron::switchOutBuffer () {
	std::swap(out_buffer, arrays.back().val);
	layers.back().changeOutput(arrays.back());
	fwd.invalidate();
	back.invalidate();
	fwd_exec.invalidate();
	back_exec.invalidate();
}
void device::multiLayerPerceptron::switchExpectBuffer () {
	std::swap(out_buffer, exp_output);
	loss.changeExpected(exp_output);
	fwd.invalidate();
	back.invalidate();
	fwd_exec.invalidate();
	back_exec.invalidate();
}

void device::multiLayerPerceptron::updateBatchSize () {
	if (batch_cnt != exec_batch_size) {
		for (device::denseLayerBatched &layer : layers) { layer.resizeBatch(batch_cnt); }
		fwd.invalidate();
		back.invalidate();
		fwd_exec.invalidate();
		back_exec.invalidate();
		exec_batch_size = batch_cnt;
	}
}

void device::multiLayerPerceptron::updateStepSize () {
	// TODO normalise with batch_sum when adding gradients to matrix with cublas to avoid precision loss
	const float next_step_size = fake_step_size / (float)batch_sum;
	if (std::abs(next_step_size - exec_step_size) > 1e-6) {
		for (device::denseLayerBatched &layer : layers) { layer.changeStepSize(next_step_size); }
		desc.invalidate();
		desc_exec.invalidate();
		exec_step_size = next_step_size;
	}
}

void device::multiLayerPerceptron::run () {
	const size_t old_batch_size = exec_batch_size;
	if (batch_cnt == 0) {
		// skip enqueuing if empty buffer
		cudaStreamSynchronize(stream);
	}
	else {
		// enqueue new run
		updateBatchSize();
		fwd.validate();
		cudaStreamSynchronize(stream);
		cudaGraphLaunch(*fwd_exec, stream);
		switchInBuffer();
		switchOutBuffer();
		batch_cnt = 0;
	}
	// get old results from output buffer
	for (size_t i = 0; i < old_batch_size; i++) {
		out_queue.emplace(out_buffer.X);
		cudaMemcpy(out_queue.back().data(),
			out_buffer + out_buffer.pitch * i,
			sizeof(float) * out_buffer.X,
			cudaMemcpyDeviceToHost);
	}
}
void device::multiLayerPerceptron::train () {
	if (batch_cnt != 0) {
		// enqueue new run
		updateBatchSize();
		back.validate();
		cudaStreamSynchronize(stream);
		cudaGraphLaunch(*back_exec, stream);
		switchInBuffer();
		switchExpectBuffer();
		batch_sum += batch_cnt;
		batch_cnt = 0;
	}
}
void device::multiLayerPerceptron::descend () {
	if (batch_sum != 0) {
		updateStepSize();
		desc.validate();
		cudaStreamSynchronize(stream);
		cudaGraphLaunch(*desc_exec, stream);
		batch_sum = 0;
	}
}

void device::multiLayerPerceptron::init (const std::vector <layerParams> &params,
                                         const device::lossFunction &loss) {
	exp_output = device::matrix <float>(params.back().size, max_batch_size);
	in_buffer = device::neuronArrayBatched(params.front().size, max_batch_size);
	out_buffer = device::matrix <float>(params.back().size, max_batch_size);
	exec_batch_size = max_batch_size;
	batch_cnt = 0;
	exec_step_size = fake_step_size;
	batch_sum = 0;
	cudaStreamCreate(&stream);

	for (const auto & param : params) {
		arrays.emplace_back(param.size, max_batch_size);
	}
	assert(getActType(params.front().act) == device::ACT_NONE);
	for (size_t i = 1; i < params.size(); i++) {
		layers.emplace_back(arrays[i - 1], arrays[i], params[i].act);
	}
	(calculateLossBatched&)loss = calculateLossBatched(arrays.back(), exp_output, loss);
}

// TODO add more safeguards to avoid reding wrong file type
// VERSION 0
void device::multiLayerPerceptron::readMetadata (std::istream &in) {
	in.seekg(0);

	const size_t metadata_size = read <size_t>(in);
	const size_t metadata_end = metadata_size + 8;

	const size_t version = read <size_t>(in);
	//assert(0 <= version);

	(size_t&)max_batch_size = read <size_t>(in);

	const size_t array_cnt = read <size_t>(in);
	std::vector <layerParams> params;
	params.resize(array_cnt);
	params[0].act = linear;
	params[0].size = read <size_t>(in);
	for (size_t i = 1; i < array_cnt; i++) {
		params[i].act = getActFunc(read <activationType>(in));
		params[i].size = read <size_t>(in);
	}
	const lossFunction loss = getLossFunc(read <lossType>(in));

	assert(in.tellg() <= metadata_end);
	in.seekg(metadata_end);

	init(params, loss);
}
bool device::multiLayerPerceptron::checkMetadata (std::istream &in) const {
	in.seekg(0);

	const size_t metadata_size = read <size_t>(in);
	const size_t metadata_end = metadata_size + 8;

	//if (0 > read <size_t>(in)) return false;
	if (read <size_t>(in) != max_batch_size) return false;

	const size_t array_cnt = read <size_t>(in);
	if (array_cnt != arrays.size()) return false;
	for (size_t i = 1; i < array_cnt - 1; i++) {
		if (read <size_t>(in) != arrays[i].getSize()) return false;
		if (read <activationType>(in) != layers[i].getType()) return false;
	}
	if (read <size_t>(in) != arrays.back().getSize()) return false;
	if (read <lossType>(in) != getLossType(loss.loss)) return false;

	assert(in.tellg() <= metadata_end);
	in.seekg(metadata_end);

	return true;
}
void device::multiLayerPerceptron::readWeights (std::istream &in) {
	for (device::denseLayerBatched &layer : layers) { layer.readWeights(in); }
}


// Public functions

void device::multiLayerPerceptron::bufferRun (const std::vector <float> &input) {
	if (batch_cnt == max_batch_size) { run(); }
	cudaMemcpy(in_buffer.val + in_buffer.val.pitch * batch_cnt,
		input.data(),
		input.size() * sizeof(float),
		cudaMemcpyHostToDevice);
	batch_cnt++;
}
void device::multiLayerPerceptron::bufferTrain (const std::vector <float> &input,
                                                       const std::vector <float> &exp_output) {
	if (batch_cnt == max_batch_size) { train(); }
	cudaMemcpy(in_buffer.val + in_buffer.val.pitch * batch_cnt,
		input.data(),
		input.size() * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(out_buffer + out_buffer.pitch * batch_cnt,
		exp_output.data(),
		exp_output.size() * sizeof(float),
		cudaMemcpyHostToDevice);
	batch_cnt++;
}
void device::multiLayerPerceptron::bufferDescent () { descend(); }

std::vector <float> device::multiLayerPerceptron::getOutput () {
	if (out_queue.empty()) {
		run();
		if (out_queue.empty()) { cudaStreamSynchronize(stream); }
	}
	std::vector <float> output = std::move(out_queue.front());
	out_queue.pop();
	return output;
}

void device::multiLayerPerceptron::reset (const unsigned long long seed) {
	for (device::denseLayerBatched &layer : layers) { layer.resetWeights(0, 1, seed); }
}
bool device::multiLayerPerceptron::load (std::istream &in) {
	if (!checkMetadata(in)) return false;
	readWeights(in);
	return true;
}
void device::multiLayerPerceptron::save (std::ostream &out) const {
	for (const device::denseLayerBatched &layer : layers) { layer.writeWeights(out); }
}
