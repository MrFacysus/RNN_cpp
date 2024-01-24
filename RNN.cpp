#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "RNN.h"

double learningRate = 0.0009;
std::vector<std::vector<double>> biases;
std::vector<std::vector<double>> errors;
std::vector<std::vector<double>> values;
std::vector<std::vector<std::vector<double>>> weights;
double lastError = 0;
std::vector<double> errorList;
std::vector<double> inputs;

void RNN::initNeurons(std::vector<int>& layers) {
	biases.resize(layers.size());
	errors.resize(layers.size());
	values.resize(layers.size());

	for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
		biases[layerIdx].resize(layers[layerIdx]);
		errors[layerIdx].resize(layers[layerIdx]);
		values[layerIdx].resize(layers[layerIdx]);
	}
}

void RNN::initWeights(std::vector<int>& layers) {
	weights.resize(layers.size() - 1);

	for (size_t layerIdx = 0; layerIdx < layers.size() - 1; layerIdx++) {
		weights[layerIdx].resize(layers[layerIdx]);

		for (size_t neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++) {
			weights[layerIdx][neuronIdx].resize(layers[layerIdx + 1], (rand() / static_cast<double>(RAND_MAX) - 0.5) * 2);
		}
	}
}

void RNN::activateTanh() {
	for (size_t layerIdx = 1; layerIdx < values.size(); layerIdx++) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			double sum = 0;
			for (size_t prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].size(); prevNeuronIdx++) {
				sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
			}
			values[layerIdx][neuronIdx] = tanh(sum + biases[layerIdx][neuronIdx]);
		}
	}
}

void RNN::activateLeakyReLU() {
	for (size_t layerIdx = 1; layerIdx < values.size(); layerIdx++) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			double sum = 0;
			for (size_t prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].size(); prevNeuronIdx++) {
				sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
			}
			values[layerIdx][neuronIdx] = sum + biases[layerIdx][neuronIdx] > 0 ? sum + biases[layerIdx][neuronIdx] : 0.01 * (sum + biases[layerIdx][neuronIdx]);
		}
	}
}

void RNN::backpropagateTanh(double* targets) {
	for (size_t neuronIdx = 0; neuronIdx < values.back().size(); neuronIdx++) {
		double derivative = 1.0 - values.back()[neuronIdx] * values.back()[neuronIdx];
		errors.back()[neuronIdx] = (targets[neuronIdx] - values.back()[neuronIdx]) * derivative;
	}

	for (size_t layerIdx = values.size() - 2; layerIdx > 0; layerIdx--) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			double sum = 0;
			for (size_t nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].size(); nextNeuronIdx++) {
				sum += weights[layerIdx][neuronIdx][nextNeuronIdx] * errors[layerIdx + 1][nextNeuronIdx];
			}

			double derivative = 1.0 - values[layerIdx][neuronIdx] * values[layerIdx][neuronIdx];
			errors[layerIdx][neuronIdx] = derivative * sum;
		}
	}

	for (size_t layerIdx = 1; layerIdx < values.size(); layerIdx++) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			for (size_t prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].size(); prevNeuronIdx++) {
				weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += learningRate * errors[layerIdx][neuronIdx] * values[layerIdx - 1][prevNeuronIdx];
			}
			biases[layerIdx][neuronIdx] += learningRate * errors[layerIdx][neuronIdx];
		}
	}
}

void RNN::backpropagateLeakyReLU(double* targets) {
	for (size_t neuronIdx = 0; neuronIdx < values.back().size(); neuronIdx++) {
		double derivative = (values.back()[neuronIdx] > 0) ? 1.0 : 0.01;
		errors.back()[neuronIdx] = (targets[neuronIdx] - values.back()[neuronIdx]) * derivative;
	}

	for (size_t layerIdx = values.size() - 2; layerIdx > 0; layerIdx--) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			double sum = 0;
			for (size_t nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].size(); nextNeuronIdx++) {
				sum += weights[layerIdx][neuronIdx][nextNeuronIdx] * errors[layerIdx + 1][nextNeuronIdx];
			}

			double derivative = (values[layerIdx][neuronIdx] > 0) ? 1.0 : 0.01;
			errors[layerIdx][neuronIdx] = sum * derivative;
		}
	}

	for (size_t layerIdx = 1; layerIdx < values.size(); layerIdx++) {
		for (size_t neuronIdx = 0; neuronIdx < values[layerIdx].size(); neuronIdx++) {
			for (size_t prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].size(); prevNeuronIdx++) {
				weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += learningRate * errors[layerIdx][neuronIdx] * values[layerIdx - 1][prevNeuronIdx];
			}
			biases[layerIdx][neuronIdx] += learningRate * errors[layerIdx][neuronIdx];
		}
	}
}


RNN::RNN(std::vector<int>& layers) {
	srand(time(NULL));
	layers[0] += layers.back();
	initNeurons(layers);
	initWeights(layers);
}

double* RNN::getTanhOutput(double* inputs) {
	for (size_t neuronIdx = 0; neuronIdx < values[0].size(); neuronIdx++) {
		values[0][neuronIdx] = inputs[neuronIdx];
	}
	activateTanh();
	return values.back().data();
}

double* RNN::getLeakyReLUOutput(double* inputs) {
	for (size_t neuronIdx = 0; neuronIdx < values[0].size(); neuronIdx++) {
		values[0][neuronIdx] = inputs[neuronIdx] > 0 ? inputs[neuronIdx] : 0.01 * inputs[neuronIdx];
	}
	activateLeakyReLU();
	return values.back().data();
}

void RNN::trainTanh(double* inputs, double* targets) {
	for (size_t neuronIdx = 0; neuronIdx < values[0].size(); neuronIdx++) {
		values[0][neuronIdx] = inputs[neuronIdx];
	}
	activateTanh();
	backpropagateTanh(targets);
}

void RNN::trainLeakyReLU(double* inputs, double* targets) {
	for (size_t neuronIdx = 0; neuronIdx < values[0].size(); neuronIdx++) {
		values[0][neuronIdx] = inputs[neuronIdx] > 0 ? inputs[neuronIdx] : 0.01 * inputs[neuronIdx];
	}
	activateLeakyReLU();
	backpropagateLeakyReLU(targets);
}
