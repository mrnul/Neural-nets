#pragma once

#include <MyHeaders\Event.h>
#include <Eigen\Dense>
#include <vector>
#include <fstream>
#include <ctime>
#undef min //i can't use std::min
#undef max //i can't use std::max

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
		vector<MatrixXf> PrevGrad;
		
		//deltas for rprop
		vector<MatrixXf> Deltas;

		//index vector to shuffle inputs
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

		//initialize weights with random numbers in range [-r, r] with r = sqrt(12 / (in + out))
		void Initialize(const vector<int> topology);
		void SetMatrices(vector<MatrixXf> & m);
		void SetGrad(vector<MatrixXf> & grad);
		void SetPrevGrad(vector<MatrixXf> & grad);
		void SetDeltas(vector<MatrixXf> & deltas);
		vector<MatrixXf> & GetMatrices();
		vector<MatrixXf> & GetGrad();
		vector<MatrixXf> & GetPrevGrad();
		vector<MatrixXf> & GetDeltas();
		void SwapGradPrevGrad();
		void ZeroGrad();
		void SetIndexVector(const vector<int> & index);
		vector<int> & GetIndexVector();
		void ShuffleIndexVector();
		//resizes and initializes index vector 0...size-1
		void ResizeIndexVector(const int size);
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

		//feed and backprop from Index[start] to Index[end - 1]
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const int start = 0, int end = 0, const float l1 = 0.0f, const float l2 = 0.0f);
		//backprop on one target
		void BackProp(const vector<float> & target);
		//update weights using this gradient
		void UpdateWeights(const float rate = 1.0f);
		//update weights using another gradient
		void UpdateWeights(const vector<MatrixXf> & grad, const float rate = 1.0f);
		//update deltas and weights
		void ResilientUpdate();

		//training with backpropagation
		void Train(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float rate,
			int batchSize = 0, const float l1 = 0.0f, const float l2 = 0.0f);
		//training with resilient backpropagation
		void TrainRPROP(const vector<vector<float>> &inputs, const vector<vector<float>> & targets);
};
