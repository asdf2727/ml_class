#pragma once

#include <vector>
#include <queue>

#include "../neurons/neuronArrayBatched.cuh"
#include "../nodes/calculateLossBatched.cuh"
#include "../nodes/denseLayerBatched.cuh"
#include "../wrappers/graphExec.cuh"

namespace device {
	class multiLayerPerceptron;
}

class device::multiLayerPerceptron {
	std::vector <device::denseLayerBatched> layers;

	device::calculateLossBatched loss;

	void buildForwardGraph (device::graph *&fwd);
	void buildBackwardGraph (device::graph *&back);
	void buildDescentGraph (device::graph *&desc);

	lazy <device::graph> fwd = lazy <device::graph>
			(std::bind(&multiLayerPerceptron::buildForwardGraph, this, std::placeholders::_1));
	lazy <device::graph> back = lazy <device::graph>
			(std::bind(&multiLayerPerceptron::buildBackwardGraph, this, std::placeholders::_1));
	lazy <device::graph> desc = lazy <device::graph>
			(std::bind(&multiLayerPerceptron::buildDescentGraph, this, std::placeholders::_1));

	void buildForwardGraphExec (device::graphExec *&fwd_exec) { fwd_exec->update(*fwd); }
	void buildBackwardGraphExec (device::graphExec *&back_exec) { back_exec->update(*back); }
	void buildDescentGraphExec (device::graphExec *&desc_exec) { desc_exec->update(*desc); }

	lazy <device::graphExec> fwd_exec = lazy <device::graphExec>
			(std::bind(&multiLayerPerceptron::buildForwardGraphExec, this, std::placeholders::_1));
	lazy <device::graphExec> back_exec = lazy <device::graphExec>
			(std::bind(&multiLayerPerceptron::buildBackwardGraphExec, this, std::placeholders::_1));
	lazy <device::graphExec> desc_exec = lazy <device::graphExec>
			(std::bind(&multiLayerPerceptron::buildDescentGraphExec, this, std::placeholders::_1));

	std::vector <device::neuronArrayBatched> arrays;
	device::matrix <float> exp_output;

	device::neuronArrayBatched in_buffer;
	device::matrix <float> out_buffer;

	void switchInBuffer ();
	void switchOutBuffer ();
	void switchExpectBuffer ();

	const size_t max_batch_size;
	size_t exec_batch_size;
	size_t batch_cnt;

	void updateBatchSize ();

	float fake_step_size;
	float exec_step_size;
	size_t batch_sum;

	void updateStepSize ();

	cudaStream_t stream;

	void run ();
	void train ();
	void descend ();

	size_t gen_count;
	size_t train_count;

	std::queue <std::vector <float>> out_queue;

public:
	struct layerParams {
		size_t size;
		device::activationFunction act;
	};

private:
	void init (const std::vector <layerParams> &params,
	           const device::lossFunction &loss);

	void readMetadata (std::istream &in);
	bool checkMetadata (std::istream &in) const;
	void readWeights (std::istream &in);

public:
	multiLayerPerceptron (const std::vector <layerParams> &params,
	                      const device::lossFunction &loss, const size_t batch_size,
	                      const float step_size = 0) :
		max_batch_size(batch_size) {
		setStepSize(step_size);
		init(params, loss);
		reset();
	}
	explicit multiLayerPerceptron (std::istream &in, const float step_size) :
		max_batch_size(0) {
		setStepSize(step_size);
		readMetadata(in);
		readWeights(in);
	}

	void setStepSize (const float step_size) { fake_step_size = step_size; }

	void bufferRun (const std::vector <float> &input);
	void bufferTrain (const std::vector <float> &input, const std::vector <float> &exp_output);
	void bufferDescent ();

	void reset (const unsigned long long seed = 0);
	bool load (std::istream &in);
	void save (std::ostream &out) const;

	std::vector <float> getOutput ();
};
