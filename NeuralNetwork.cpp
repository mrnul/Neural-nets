#include <MyHeaders\NeuralNetwork.h>
#include <iostream>

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
	const float coeff = b - a;
	for (int i = 0; i < size; i++)
		vec[i] = coeff * (vec[i] - min) / denom + a;
}

NeuralNetwork::NeuralNetwork()
{
	std::srand((unsigned int)std::time(0));
}

NeuralNetwork::NeuralNetwork(const vector<int> topology)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology);
}

float NeuralNetwork::Activation(const float x)
{
	//ELU for hidden layers
	return x >= 0 ? x : exp(x) - 1;
}

float NeuralNetwork::Derivative(const float x)
{
	return x >= 0 ? 1 : exp(x);
}

float NeuralNetwork::OutActivation(const float x)
{
	//logistic for output
	return 1.0f / (1.0f + exp(-x));
}

float NeuralNetwork::OutDerivative(const float x)
{
	const float af = OutActivation(x);
	return af * (1.0f - af);
}

void NeuralNetwork::Initialize(const vector<int> topology)
{
	const int numOfLayers = topology.size();
	const int lastIndex = numOfLayers - 1;

	D.resize(numOfLayers);
	Ex.resize(numOfLayers);
	O.resize(numOfLayers);
	Matrices.resize(numOfLayers);
	Grad.resize(numOfLayers);
	PrevGrad.resize(numOfLayers);
	Deltas.resize(numOfLayers);

	//initialize the rest beginning from the second layer
	for (int i = 1; i < numOfLayers; i++)
	{
		//these don't need a bias node
		D[i].setZero(topology[i]);
		Ex[i].setZero(topology[i]);

		//these need a bias node
		O[i - 1].setZero(topology[i - 1] + 1);
		O[i - 1][topology[i - 1]] = 1;

		//+1 for the bias of the prev layer
		//initialize Matrices with random numbers
		Matrices[i].setRandom(topology[i - 1] + 1, topology[i]);
		Matrices[i] *= sqrt(12.0f / (topology[i - 1] + 1.0f + topology[i]));

		//initialize Grad with value 0 and deltas
		Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);
		Deltas[i].setConstant(topology[i - 1] + 1, topology[i], 0.0125f);
	}

	//last layer does not have a bias node
	O[lastIndex].setZero(topology[lastIndex]);
}

void NeuralNetwork::SetMatrices(vector<MatrixXf> & m)
{
	Matrices = m;
}

void NeuralNetwork::SetGrad(vector<MatrixXf> & grad)
{
	Grad = grad;
}

void NeuralNetwork::SetPrevGrad(vector<MatrixXf> & grad)
{
	PrevGrad = grad;
}

void NeuralNetwork::SetDeltas(vector<MatrixXf>& deltas)
{
	Deltas = deltas;
}

vector<MatrixXf>& NeuralNetwork::GetMatrices()
{
	return Matrices;
}

vector<MatrixXf>& NeuralNetwork::GetGrad()
{
	return Grad;
}

vector<MatrixXf>& NeuralNetwork::GetPrevGrad()
{
	return PrevGrad;
}

vector<MatrixXf>& NeuralNetwork::GetDeltas()
{
	return Deltas;
}

void NeuralNetwork::SwapGradPrevGrad()
{
	std::swap(Grad, PrevGrad);
}

void NeuralNetwork::ZeroGrad()
{
	const int MatricesSize = Matrices.size();

	for (int l = 1; l < MatricesSize; l++)
		Grad[l].setZero();
}

void NeuralNetwork::SetIndexVector(const vector<int>& index)
{
	Index = index;
}

vector<int>& NeuralNetwork::GetIndexVector()
{
	return Index;
}

void NeuralNetwork::ShuffleIndexVector()
{
	for (int i = Index.size() - 1; i > 0; i--)
		std::swap(Index[i], Index[rand() % (i + 1)]);
}

void NeuralNetwork::ResizeIndexVector(const int size)
{
	Index.resize(size);
	for (int i = 0; i < size; i++)
		Index[i] = i;
}

MatrixXf & NeuralNetwork::operator[](int layer)
{
	return Matrices[layer];
}

const RowVectorXf & NeuralNetwork::FeedForward(const vector<float> & input)
{
	std::copy(input.data(), input.data() + input.size(), O[0].data());

	const int lastIndex = Matrices.size() - 1;
	for (int m = 1; m < lastIndex; m++)
	{
		Ex[m].noalias() = O[m - 1] * Matrices[m];
		O[m].head(O[m].size() - 1) = Ex[m].unaryExpr(&Activation);
	}

	Ex[lastIndex].noalias() = O[lastIndex - 1] * Matrices[lastIndex];
	O[lastIndex] = Ex[lastIndex].unaryExpr(&OutActivation);

	return O.back();
}

const RowVectorXf & NeuralNetwork::Evaluate(const vector<float> & input)
{
	std::copy(input.data(), input.data() + input.size(), O[0].data());

	const int lastIndex = Matrices.size() - 1;
	for (int m = 1; m < lastIndex; m++)
	{
		O[m].head(O[m].size() - 1).noalias() = O[m - 1] * Matrices[m];
		O[m].head(O[m].size() - 1) = O[m].head(O[m].size() - 1).unaryExpr(&Activation);
	}

	O[lastIndex].noalias() = O[lastIndex - 1] * Matrices[lastIndex];
	O[lastIndex] = O[lastIndex].unaryExpr(&OutActivation);

	return O.back();
}

float NeuralNetwork::SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff)
{
	const int inputCount = inputs.size();
	float ret = 0;
	for (int i = 0; i < inputCount; i++)
	{
		Evaluate(inputs[i]);

		const int resSize = O.back().size();

		float error = 0;
		for (int k = 0; k < resSize; k++)
			error += (targets[i][k] - O.back()[k]) * (targets[i][k] - O.back()[k]);

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

float NeuralNetwork::Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	int correct = 0;
	const int inputSize = inputs.size();
	for (int i = 0; i < inputSize; i++)
	{
		Evaluate(inputs[i]);
		bool allCorrect = true;
		const int outSize = O.back().size();
		for (int r = 0; r < outSize; r++)
		{
			if ((targets[i][r] > 0.5 && O.back()[r] < 0.5) ||
				(targets[i][r] < 0.5 && O.back()[r] > 0.5))
			{
				allCorrect = false;
				break;
			}
		}

		if (allCorrect)
			correct++;
	}

	return correct * 100.0f / inputs.size();
}

bool NeuralNetwork::WriteWeightsToFile(const char * path) const
{
	std::ofstream file(path, std::ios::out | std::ios::binary);
	if (!file.is_open())
		return false;

	const int matricesCount = Matrices.size();
	for (int m = 1; m < matricesCount; m++)
	{
		const int curSize = Matrices[m].size() * sizeof(float);
		file.write((const char*)Matrices[m].data(), curSize);
	}

	return true;
}

bool NeuralNetwork::LoadWeightsFromFile(const char * path)
{
	std::ifstream file(path, std::ios::in | std::ios::binary);
	if (!file.is_open())
		return false;

	const int matricesCount = Matrices.size();
	for (int m = 1; m < matricesCount; m++)
	{
		const int curSize = Matrices[m].size() * sizeof(float);
		file.read((char*)Matrices[m].data(), curSize);
	}

	return true;
}

void NeuralNetwork::FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const int start, int end, const float l1, const float l2)
{
	if (end == 0)
		end = inputs.size();

	for (int i = start; i < end; i++)
	{
		FeedForward(inputs[Index[i]]);
		BackProp(targets[Index[i]]);
	}

	//add l1 and l2 regularization terms to the gradient
	//Grad = Grad + l1term + l2term
	const int lCount = Grad.size();

	//both l1 and l2
	if (l1 != 0.0f && l2 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l] += l1 * Matrices[l].unaryExpr(&Sign) + l2 * Matrices[l];
	}
	//only l1
	else if (l1 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l] += l1 * Matrices[l].unaryExpr(&Sign);
	}
	//only l2
	else if (l2 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l] += l2 * Matrices[l];
	}
}

void NeuralNetwork::BackProp(const vector<float> & target)
{
	const int L = D.size() - 1;
	const int outSize = O[L].size();

	//delta for output
	for (int j = 0; j < outSize; j++)
		D[L][j] = 2 * OutDerivative(Ex[L][j]) * (O[L][j] - target[j]);

	//grad for output
	Grad[L].noalias() += O[L - 1].transpose() * D[L];

	//delta for hidden
	for (int l = L; l > 1; l--)
	{
		//calc the sums
		D[l - 1].noalias() = Matrices[l].topRows(D[l - 1].size()) * D[l].transpose();

		//multiply with derivatives
		D[l - 1].array() *= Ex[l - 1].unaryExpr(&Derivative).array();

		//grad for hidden
		Grad[l - 1].noalias() += O[l - 2].transpose() * D[l - 1];
	}
}

void NeuralNetwork::UpdateWeights(const float rate)
{
	const int MatricesSize = Matrices.size();
	for (int l = 1; l < MatricesSize; l++)
		Matrices[l] -= rate * Grad[l];
}

void NeuralNetwork::UpdateWeights(const vector<MatrixXf> & grad, const float rate)
{
	const int MatricesSize = Matrices.size();
	for (int l = 1; l < MatricesSize; l++)
		Matrices[l] -= rate * grad[l];
}

void NeuralNetwork::ResilientUpdate()
{
	const int mSize = Matrices.size();

	for (int m = 0; m < mSize; m++)
	{
		const int wSize = Matrices[m].size();
		for (int w = 0; w < wSize; w++)
		{
			const int sign = Sign(Grad[m](w) * PrevGrad[m](w));
			if (sign > 0)
			{
				Deltas[m](w) = std::min(Deltas[m](w) * 1.2f, 50.0f);
			}
			else if (sign < 0)
			{
				Deltas[m](w) = std::max(Deltas[m](w) * 0.5f, 1e-6f);
				Grad[m](w) = 0.0f;
			}

			Matrices[m](w) -= Sign(Grad[m](w)) * Deltas[m](w);
		}
	}
}

void NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate,
	int batchSize, const float l1, const float l2)
{
	const int inputSize = inputs.size();

	if (batchSize == 0 || batchSize > inputSize)
		batchSize = inputSize;

	//resize if needed
	if (Index.size() != inputSize)
		ResizeIndexVector(inputSize);

	//shuffle index vector if needed
	if (batchSize != inputSize)
		ShuffleIndexVector();

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + batchSize, inputSize);

		FeedAndBackProp(inputs, targets, start, end, l1, l2);
		UpdateWeights(rate);

		SwapGradPrevGrad();
		ZeroGrad();
	}
}


void NeuralNetwork::TrainRPROP(const vector<vector<float>>& inputs, const vector<vector<float>>& targets)
{
	const int inputSize = inputs.size();

	//resize if needed
	if (Index.size() != inputSize)
		ResizeIndexVector(inputSize);

	FeedAndBackProp(inputs, targets);

	ResilientUpdate();

	SwapGradPrevGrad();
	ZeroGrad();
}