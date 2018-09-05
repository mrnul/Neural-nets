#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <fstream>
#include <ctime>

class NeuralNetwork
{
	private:
		neuralnetworkbase::NNBase Base;

	public:

		neuralnetworkbase::NNParams Params;

		NeuralNetwork();
		NeuralNetwork(const vector<int> topology, const int ThreadCount = 1);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<int> topology, const int ThreadCount = 1);

		//returns Base.O.back()
		const MatrixXf & Evaluate(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char * path) const;
		bool LoadWeightsFromFile(const char * path);

		//training with backpropagation, using Params member
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
};
