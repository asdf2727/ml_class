cmake_minimum_required(VERSION 3.29)
project(ml_class CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ml_class
		src/main.cu
		src/device/wrappers/array.cuh
		src/device/wrappers/matrix.cuh
		src/device/device.cuh
		src/device/nodes/linearLayerBatched.cuh
		src/device/neurons/neuronArray.cuh
		src/device/neurons/neuronArrayBatched.cuh
		src/device/nodes/activations/sigmoid.cuh
		src/device/networks/multiLayerPerceptron.cuh
		src/device/nodes/abstract/node.cuh
		src/device/nodes/denseLayerBatched.cuh
		src/device/nodes/abstract/weightedNode.cuh
		src/device/nodes/activateLayerBatched.cuh
		src/device/nodes/abstract/nodeBatched.cuh
		src/device/nodes/abstract/weightedNodeBatched.cuh
		src/device/nodes/abstract/IONodeBatched.cuh
		src/device/nodes/calculateLossBatched.cuh
		src/device/nodes/losses/meanSquaredError.cuh
		src/device/nodes/losses/BinaryCrossEntropy.cuh
		src/device/nodes/activations/rectifiedLinearUnit.cuh
		src/lazy.cuh
		src/device/wrappers/graph.cuh
		src/link.cuh
)

set_target_properties(ml_class PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON)
