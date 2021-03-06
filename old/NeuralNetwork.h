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
		vector<RowVectorXf> Ex;

		//each neuron's output
		vector<RowVectorXf> O;

		//each neuron's delta
		vector<RowVectorXf> D;

		//the gradient
		vector<MatrixXf> Grad;
		vector<MatrixXf> PrevGrad;

		//index vector to shuffle inputs
		vector<int> Index;

	public:

		NNParams Params;

		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<unsigned int> topology);
		//initialize everything except for the weights
		void InitializeNoWeights(const vector<unsigned int> topology);
		vector<MatrixXf> & GetMatrices();
		vector<MatrixXf> & GetGrad();
		vector<MatrixXf> & GetPrevGrad();
		void NormalizeGrad();
		void SwapGradPrevGrad();
		void ZeroGrad();
		vector<int> & GetIndexVector();
		void ShuffleIndexVector();
		//resizes and initializes index vector 0...size-1
		void ResizeIndexVector(const int size);

		//returns O.back()
		const RowVectorXf & FeedForward(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		//feed and backprop from inputs[Index[start]] to inputs[Index[end - 1]], using *this* weights and index vector
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const int start, int end, const float DropOutRate);
		//feed and backprop using other weights and index vector
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const vector<MatrixXf> & matrices, const vector<int> & index,
			const int start, int end, const float DropOutRate);

		//backprop on one target
		void BackProp(const vector<float> & target);
		//Wnew = Wold + rate * Grad
		void UpdateWeights(const float rate);
		//Grad += L1term + L2term
		void AddL1L2(const float l1, const float l2);
		//Grad += PrevGrad * momentum
		void AddMomentum(const float momentum);
		//checks if all weights are finite (no nan or inf)
		bool AllFinite();

		//training with backpropagation, using Params member
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
};
