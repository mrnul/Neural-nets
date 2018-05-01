#pragma once

#include <Eigen\Dense>
#include <vector>
#include <fstream>
#include <ctime>

using std::vector;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;


//each element is in range [a,b]
void NormalizeVector(vector<float> & vec, const float a = 0.1f, const float b = 0.9f);
//each column is scaled in [a,b]
void NormalizeColumnwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
//each row is scaled in [a,b]
void NormalizeRowwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
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

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<int> topology);
		void SetMatrices(const vector<MatrixXf> & m);
		void SetGrad(const vector<MatrixXf> & grad);
		void SetPrevGrad(const vector<MatrixXf> & grad);
		vector<MatrixXf> & GetMatrices();
		vector<MatrixXf> & GetGrad();
		vector<MatrixXf> & GetPrevGrad();
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

		//uses only member O
		const RowVectorXf & Evaluate(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		//feed and backprop from Index[start] to Index[end - 1]
		void FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
			const int start = 0, int end = 0);
		//backprop on one target
		void BackProp(const vector<float> & target);
		//update weights using this gradient
		void UpdateWeights(const float rate);
		//update weights using another gradient
		void UpdateWeights(const vector<MatrixXf> & grad, const float rate);
		//add l1 and l2 regularization terms to the gradient
		void AddL1L2(const float l1, const float l2);
		//add momentum to the gradient
		void AddMomentum(const float momentum);
		//checks if all weights are finite (no nan or inf)
		bool AllFinite();

		//training with backpropagation
		void Train(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float rate,
			const float momentum = 0.0f, int batchSize = 0, const float l1 = 0.0f, const float l2 = 0.0f);
};
