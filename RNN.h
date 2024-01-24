#pragma once

#include <vector>

class RNN {
public:
	RNN(std::vector<int>& layers);
	double* getTanhOutput(double* inputs);
	double* getLeakyReLUOutput(double* inputs);
	void backpropagateTanh(double* targets);
	void backpropagateLeakyReLU(double* targets);
	void trainTanh(double* inputs, double* targets);
	void trainLeakyReLU(double* inputs, double* targets);
private:
	void initNeurons(std::vector<int>& layers);
	void initWeights(std::vector<int>& layers);
	void activateTanh();
	void activateLeakyReLU();
};
