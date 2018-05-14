#pragma once

#include <Eigen\Dense>
#include <vector>

using std::vector;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;

typedef float(*ActivationFunction)(const float);
typedef float(*ActivationDerivative)(const float);

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
}

//each element is in range [a,b]
void NormalizeVector(vector<float> & vec, const float a = 0.1f, const float b = 0.9f);
//each column is scaled in [a,b]
void NormalizeColumnwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
//each row is scaled in [a,b]
void NormalizeRowwise(vector<vector<float>> & data, const float a = 0.1f, const float b = 0.9f);
inline int Sign(const float x) { return (x > 0.0f) - (x < 0.0f); }

//returns the output o.back()
const RowVectorXf & NNFeedForward(const vector<float> & input, const vector<MatrixXf> & matrices, vector<RowVectorXf> & ex, vector<RowVectorXf> & o);

//finds the gradient
void NNBackProp(const vector<float> & target, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad,
	const vector<RowVectorXf> & ex, const vector<RowVectorXf> & o, vector<RowVectorXf> & d);

//goes from index[start] to index[end - 1]
void NNFeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const vector<MatrixXf> & matrices, vector<MatrixXf> & grad, const vector<int> & index,
	vector<RowVectorXf> & ex, vector<RowVectorXf> & o, vector<RowVectorXf> & d,
	const int start, int end);