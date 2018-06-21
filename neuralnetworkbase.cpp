#include <MyHeaders/neuralnetworkbase.h>

void NormalizeVector(vector<float> & vec, const float a, const float b)
{
	float min = vec[0];
	float max = vec[0];
	const int size = vec.size();
	for (int i = 1; i < size; i++)
	{
		if (min > vec[i])
			min = vec[i];
		else if (max < vec[i])
			max = vec[i];
	}

	const float denom = max - min;
	if (denom == 0.0f)
	{
		if (max > b)
		{
			for (int i = 0; i < size; i++)
				vec[i] = b;
		}
		else if (max < a)
		{
			for (int i = 0; i < size; i++)
				vec[i] = a;
		}

		return;
	}
	
	const float coeff = b - a;
	for (int i = 0; i < size; i++)
		vec[i] = coeff * (vec[i] - min) / denom + a;
}

void NormalizeColumnwise(vector<vector<float>> & data, const float a, const float b)
{
	const int DataCount = data.size();
	const int FeatureCount = data[0].size();

	for (int f = 0; f < FeatureCount; f++)
	{
		float min = data[0][f];
		float max = data[0][f];
		for (int d = 1; d < DataCount; d++)
		{
			if (min > data[d][f])
				min = data[d][f];
			else if (max < data[d][f])
				max = data[d][f];
		}

		const float denom = max - min;
		if (denom == 0.0f)
		{
			if (max > b)
			{
				for (int d = 0; d < DataCount; d++)
					data[d][f] = b;
			}
			else if (max < a)
			{
				for (int d = 0; d < DataCount; d++)
					data[d][f] = a;
			}
			continue;
		}

		const float coeff = b - a;
		for (int d = 0; d < DataCount; d++)
			data[d][f] = coeff * (data[d][f] - min) / denom + a;
	}

}

void NormalizeRowwise(vector<vector<float>> & data, const float a, const float b)
{
	const int size = data.size();
	for (int i = 0; i < size; i++)
		NormalizeVector(data[i], a, b);
}

const RowVectorXf & NNFeedForward(const vector<float> & input, const vector<MatrixXf> & matrices, vector<RowVectorXf> & ex, vector<RowVectorXf> & o)
{
	std::copy(input.data(), input.data() + input.size(), o[0].data());

	const int lastIndex = matrices.size() - 1;
	for (int m = 1; m < lastIndex; m++)
	{
		ex[m].noalias() = o[m - 1] * matrices[m];
		o[m].head(o[m].size() - 1) = ex[m].unaryExpr(&NNFunctions::ELU);
	}

	ex[lastIndex].noalias() = o[lastIndex - 1] * matrices[lastIndex];
	o[lastIndex] = ex[lastIndex].unaryExpr(&NNFunctions::Linear);

	return o.back();
}

void NNBackProp(const vector<float> & target, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad,
	const vector<RowVectorXf> & ex, const vector<RowVectorXf> & o, vector<RowVectorXf> & d)
{
	const int L = d.size() - 1;
	const int outSize = o[L].size();

	//delta for output
	for (int j = 0; j < outSize; j++)
		d[L][j] = NNFunctions::LinearDerivative(ex[L][j]) * (o[L][j] - target[j]);

	//grad for output
	grad[L].noalias() += o[L - 1].transpose() * d[L];

	//delta for hidden
	for (int l = L; l > 1; l--)
	{
		//calc the sums
		d[l - 1].noalias() = matrices[l].topRows(d[l - 1].size()) * d[l].transpose();

		//multiply with derivatives
		d[l - 1].array() *= ex[l - 1].unaryExpr(&NNFunctions::ELUDerivative).array();

		//grad for hidden
		grad[l - 1].noalias() += o[l - 2].transpose() * d[l - 1];
	}
}

void NNFeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const vector<MatrixXf> & matrices, vector<MatrixXf> & grad, const vector<int> & index,
	vector<RowVectorXf> & ex, vector<RowVectorXf> & o, vector<RowVectorXf> & d,
	const int start, const int end)
{
	for (int i = start; i < end; i++)
	{
		NNFeedForward(inputs[index[i]], matrices, ex, o);
		NNBackProp(targets[index[i]], matrices, grad, ex, o, d);
	}
}
