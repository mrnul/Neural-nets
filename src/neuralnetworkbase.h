#pragma once

#include <Eigen\Core>
#include <vector>
#include <fstream>
#include <ctime>
#include <map>

using std::map;
using std::vector;
using std::ifstream;
using Eigen::MatrixXf;

namespace neuralnetworkbase
{
	//Basic structure with the necessary functions
	struct NNBase
	{
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

		void InitializeBase(vector<int> topology, const int threadCount = 0);
		void InitializeIndexVector(const int size);
		void ShuffleIndexVector();
		void SwapAndZeroGrad();

		bool WriteWeightsToFile(const char * path) const;
		bool LoadWeightsFromFile(const char * path);

		void Dropout(MatrixXf & layer, const float DropOutRate);
		void AddL1L2(const float l1, const float l2);
		void AddMomentum(const float momentum);
		void UpdateWeights(const float rate, const bool NormalizeGrad);

		//returns the output o.back()
		const MatrixXf & FeedForward(const vector<float> & input, const float DropOutRate);
		void Backprop(const vector<float> & target);
	};

	struct NNParams
	{
		float L1;
		float L2;
		float Momentum;
		float LearningRate;
		float DropOutRate;
		int BatchSize;
		bool NormalizeGradient;

		NNParams() : L1(0), L2(0), Momentum(0), LearningRate(0.001f), DropOutRate(0), BatchSize(1), NormalizeGradient(false) {}
	};

	//Activation functions and their derivatives
	namespace functions
	{
		inline float ELU(const float x)
		{
			return x >= 0 ? x : exp(x) - 1;
		}

		inline float ELUDerivative(const float x)
		{
			return x >= 0 ? 1 : exp(x);
		}

		inline float Linear(const float x)
		{
			return x;
		}

		inline float LinearDerivative(const float x)
		{
			return 1.f;
		}

		inline float Logistic(const float x)
		{
			return 1.0f / (1.0f + exp(-x));
		}

		inline float LogisticDerivative(const float x)
		{
			const float af = Logistic(x);
			return af * (1.0f - af);
		}

		inline float Softmax(const float x, const MatrixXf & v)
		{
			const float C = v.maxCoeff();
			return exp(x - C) / v.unaryExpr([&](const float x) { return exp(x - C); }).sum();
		}

		inline float SoftmaxDerivative(const float x, const MatrixXf & v)
		{
			const float sm = Softmax(x, v);
			return sm * (1.f - sm);
		}
	}

	//each element is scaled in [a,b]
	void NormalizeVector(vector<float> & vec, const float a = 0.1f, const float b = 0.9f);
	//each column is scaled in [a,b]
	void NormalizeColumnwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
	//each row is scaled in [a,b]
	void NormalizeRowwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
	//Result Vector = (Input Vector - Mean) / Variance
	void StandarizeVector(vector<float> & vec);
	//for each row: Result Row = (Input Row - Mean) / Variance
	void StandarizeVectors(vector<vector<float>> & data);
	inline int Sign(const float x) { return (x > 0.0f) - (x < 0.0f); }
	//loads all data, creates maps for encoding and decoding, returns false if file could not open
	bool OneHotEncDec(const char * path, map<unsigned char, int> & enc, map<int, unsigned char> & dec, vector<__int16> & data);
	//returns the index of the max element
	int IndexOfMax(const MatrixXf & v);
	int IndexOfMax(const vector<float> & v);
}