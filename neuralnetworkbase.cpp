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

void StandarizeVector(vector<float> & vec)
{
	float mean = 0;
	float sd = 0;
	const int size = vec.size();
	
	for (int i = 0; i < size; i++)
		mean += vec[i];
	mean /= size;

	for (int i = 0; i < size; i++)
		sd += (vec[i] - mean) * (vec[i] - mean);
	sd /= size - 1;
	sd = sqrt(sd);

	for (int i = 0; i < size; i++)
		vec[i] = (vec[i] - mean) / sd;
}

void StandarizeVectors(vector<vector<float>> & data)
{
	const int size = data.size();
	for (int i = 0; i < size; i++)
		StandarizeVector(data[i]);
}

void NNDropOut(RowVectorXf & o, const float DropOutRate)
{
	if (DropOutRate == 0.f)
		return;

	//-1 to ignore last neuron (don't dropout the bias)
	const auto N = o.size() - 1;
	for (auto n = 0; n < N; n++)
	{
		const float rnd = (float)rand() / RAND_MAX;
		if (rnd <= DropOutRate)
			o(n) = 0.f;
	}
}

void NNAddMomentum(const float momentum, vector<MatrixXf>& grad, const vector<MatrixXf>& prevgrad)
{
	if (momentum == 0.f)
		return;

	const int lCount = grad.size();
	for (int l = 0; l < lCount; l++)
		grad[l] += momentum * prevgrad[l];
}

void NNAddL1L2(const float l1, const float l2, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad)
{
	//Grad = Grad + l1term + l2term
	//don't regularize the bias
	//topRows(Grad[l].rows() - 1) skips the last row (the biases)

	const int lCount = grad.size();

	//both l1 and l2
	if (l1 != 0.f && l2 != 0.f)
	{
		for (int l = 1; l < lCount; l++)
			grad[l].topRows(grad[l].rows() - 1) +=
			l1 * matrices[l].topRows(grad[l].rows() - 1).unaryExpr(&Sign)
			+ l2 * matrices[l].topRows(grad[l].rows() - 1);
	}
	//only l1
	else if (l1 != 0.f)
	{
		for (int l = 1; l < lCount; l++)
			grad[l].topRows(grad[l].rows() - 1) += l1 * matrices[l].topRows(grad[l].rows() - 1).unaryExpr(&Sign);
	}
	//only l2
	else if (l2 != 0.f)
	{
		for (int l = 1; l < lCount; l++)
			grad[l].topRows(grad[l].rows() - 1) += l2 * matrices[l].topRows(grad[l].rows() - 1);
	}
}

const RowVectorXf & NNFeedForward(const vector<float> & input, const vector<MatrixXf> & matrices,
	vector<RowVectorXf> & ex, vector<RowVectorXf> & o, const float DropOutRate)
{
	std::copy(input.data(), input.data() + input.size(), o[0].data());
	
	NNDropOut(o[0], DropOutRate);

	const int lastIndex = matrices.size() - 1;
	for (int m = 1; m < lastIndex; m++)
	{
		ex[m].noalias() = o[m - 1] * matrices[m];
		o[m].head(o[m].size() - 1) = ex[m].unaryExpr(&NNFunctions::ELU);
		
		NNDropOut(o[m], DropOutRate);
	}

	ex[lastIndex].noalias() = o[lastIndex - 1] * matrices[lastIndex];
	o[lastIndex] = ex[lastIndex].unaryExpr([&](const float x) { return NNFunctions::Softmax(x, ex[lastIndex]); });

	return o.back();
}

void NNBackProp(const vector<float> & target, const vector<MatrixXf> & matrices, vector<MatrixXf> & grad,
	const vector<RowVectorXf> & ex, const vector<RowVectorXf> & o, vector<RowVectorXf> & d)
{
	const int L = d.size() - 1;
	const int outSize = o[L].size();

	//delta for output
	for (int j = 0; j < outSize; j++)
		d[L][j] = (o[L][j] - target[j]);

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
	const int start, const int end, const float DropOutRate)
{
	for (int i = start; i < end; i++)
	{
		NNFeedForward(inputs[index[i]], matrices, ex, o, DropOutRate);
		NNBackProp(targets[index[i]], matrices, grad, ex, o, d);
	}
}
