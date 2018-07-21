#pragma once

#include <Eigen\Dense>
#include <vector>

using std::vector;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;

namespace NNFunctions
{
	inline float ELU(const float x)
	{
		return x >= 0 ? x : exp(x) - 1;
	}

	inline float ELUDerivative(const float x)
	{
		return x >= 0 ? 1 : exp(x);
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
	inline float Linear(const float x)
	{
		return x;
	}
	inline float LinearDerivative(const float x)
	{
		return 1.0f;
	}
	inline float Softmax(const float x, const RowVectorXf & v)
	{
		const float C = v.maxCoeff();
		return exp(x - C) / v.unaryExpr([&](const float x) { return exp(x - C); }).sum();
	}
	inline float SoftmaxDerivative(const float x, const RowVectorXf & v)
	{
		const float sm = Softmax(x, v);
		return sm * (1 - sm);
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

/*
matrices : the weights of the neural network
ex       : the excitation of each neuron (weighted sum)
o        : the output of each neuron
d        : delta of each neuron
grad     : the gradient
prevgrad : the previous gradient
*/

void NNDropOut(RowVectorXf & o, const float DropOutRate);
void NNAddL1L2(const float l1, const float l2, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad);
void NNAddMomentum(const float momentum, vector<MatrixXf> & grad, const vector<MatrixXf> & prevgrad);

//returns the output o.back()
const RowVectorXf & NNFeedForward(const vector<float> & input, const vector<MatrixXf> & matrices,
	vector<RowVectorXf> & ex, vector<RowVectorXf> & o, const float DropOutRate);

//finds the gradient
void NNBackProp(const vector<float> & target, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad,
	const vector<RowVectorXf> & ex, const vector<RowVectorXf> & o, vector<RowVectorXf> & d);

//goes from inputs[index[start]] to inputs[index[end - 1]]
void NNFeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const vector<MatrixXf> & matrices, vector<MatrixXf> & grad, const vector<int> & index,
	vector<RowVectorXf> & ex, vector<RowVectorXf> & o, vector<RowVectorXf> & d,
	const int start, const int end, const float DropOutRate);
