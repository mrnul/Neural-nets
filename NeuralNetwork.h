#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <fstream>
#include <ctime>

class NeuralNetwork
{
	private:
		//each layer's matrix
		vector<MatrixXf> Matrices;

		//each neuron's excitation
		vector<MatrixXf> Ex;

		//each neuron's output
		vector<MatrixXf> O;

		//each neuron's delta
		vector<MatrixXf> D;

		//the gradient
		vector<MatrixXf> Grad;
		vector<MatrixXf> PrevGrad;

		//index vector to shuffle inputs
		vector<int> Index;

	public:

		NNParams Params;

		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology, const int ThreadCount = 1);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<unsigned int> topology, const int ThreadCount = 1);
		vector<MatrixXf> & GetMatrices();
		vector<MatrixXf> & GetGrad();
		vector<MatrixXf> & GetPrevGrad();
		void SwapGradPrevGrad();
		void ZeroGrad();
		vector<int> & GetIndexVector();
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

		//feed and backprop from inputs[Index[start]] to inputs[Index[end - 1]], using *this* weights and index vector
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const int start, int end, const float DropOutRate);

		//backprop on one target
		void BackProp(const vector<float> & target);
		//Wnew = Wold + rate * Grad
		void UpdateWeights();
		//Grad += L1term + L2term
		void AddL1L2();
		//Grad += PrevGrad * momentum
		void AddMomentum();
		//checks if all weights are finite (no nan or inf)
		bool AllFinite();

		//training with backpropagation, using Params member
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
};
