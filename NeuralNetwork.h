#pragma once

#include <MyHeaders\Event.h>
#include <Eigen\Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>

using std::vector;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;


//for each i, vec[i] is in range [a,b]
void NormalizeVector(vector<float> & vec, const float a = 0, const float b = 1);
inline int Sign(const float x) { return (x > 0.0f) - (x < 0.0f); }

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

		//to shuffle inputs
		vector<int> Index;
	public:
		NeuralNetwork();
		NeuralNetwork(const vector<int> topology);

		//activation and derivative of hidden layers
		static float Activation(const float x);
		static float Derivative(const float x);

		//activation and derivative of output layer
		static float OutActivation(const float x);
		static float OutDerivative(const float x);

		void Initialize(const vector<int> topology);
		void SetMatrices(const vector<MatrixXf> & m);
		const vector<MatrixXf> & GetMatrices() const;
		const vector<MatrixXf> & GetGrad() const;
		void ZeroGrad();
		void SetIndex(const vector<int> & index);
		const vector<int> & GetIndex() const;
		void ShuffleIndex();
		void ResizeIndex(const int size);
		MatrixXf & operator[](int layer);

		//uses member Ex and O
		const RowVectorXf & FeedForward(const vector<float> & input);

		//only uses member O
		const RowVectorXf & Evaluate(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		void FeedAndBackProp(const int start, const int end, const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		void UpdateWeights(const float rate);
		void UpdateWeights(const vector<MatrixXf> & grad, const float rate);

		//training with backpropagation
		void Train(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float rate, int batchSize = 0);
};
