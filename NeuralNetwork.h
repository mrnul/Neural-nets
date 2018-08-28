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
		NeuralNetwork(const vector<unsigned int> topology, const int ThreadCount = 1);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<unsigned int> topology, const int ThreadCount = 1);
		void SwapGradPrevGrad();
		void ZeroGrad();
		void ShuffleIndexVector();
		//resizes and initializes index vector 0...size-1
		void ResizeIndexVector(const int size);

		//returns O.back()
		const MatrixXf & FeedForward(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		//feed and backprop from inputs[Index[start]] to inputs[Index[end - 1]]
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const int start, int end);

		//backprop on one target
		void BackProp(const vector<float> & target);
		//Wnew = Wold + rate * Grad
		void UpdateWeights();
		//Grad += L1term + L2term
		void AddL1L2();
		//Grad += PrevGrad * momentum
		void AddMomentum();

		//training with backpropagation, using Params member
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
};
