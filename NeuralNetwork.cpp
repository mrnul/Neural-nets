#include <MyHeaders\NeuralNetwork.h>

void NormalizeVector(vector<float> & vec, const float a, const float b)
{
	float min = vec[0];
	float max = vec[0];

	for (unsigned int i = 1; i < vec.size(); i++)
	{
		if (min > vec[i])
			min = vec[i];
		else if (max < vec[i])
			max = vec[i];
	}

	const float denom = max - min;
	const float coeff = b - a;
	for (unsigned int i = 0; i < vec.size(); i++)
		vec[i] = coeff * (vec[i] - min) / denom + a;
}

int Sign(const float x)
{
	if (x > 0)
		return 1;
	if (x < 0)
		return -1;

	return 0;
}

NeuralNetwork::NeuralNetwork()
{
	std::srand((unsigned int)std::time(0));
}

NeuralNetwork::NeuralNetwork(const vector<unsigned int> topology)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology);
}

void NeuralNetwork::Initialize(const vector<unsigned int> topology)
{
	const unsigned int numOfLayers = topology.size();
	const unsigned int lastIndex = numOfLayers - 1;

	D.resize(numOfLayers);
	Ex.resize(numOfLayers);
	O.resize(numOfLayers);
	Matrices.resize(numOfLayers);
	Grad.resize(numOfLayers);
	
	//these two are for rprop
	PrevGrad.resize(numOfLayers);
	Delta.resize(numOfLayers);

	//initialize the rest beginning from the second layer
	for (unsigned int i = 1; i < numOfLayers; i++)
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
		Matrices[i] *= sqrt(12.0 / (topology[i - 1] + 1.0 + topology[i]));

		//initialize Grads with value 0
		Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);

		//initialize Delta
		Delta[i].setConstant(topology[i - 1] + 1, topology[i], 0.0125);
	}

	//last layer does not have a bias node
	O[lastIndex].setZero(topology[lastIndex]);
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
	return 1.0 / (1.0 + exp(-x));
}

float NeuralNetwork::OutDerivative(const float x)
{
	const float af = OutActivation(x);
	return af * (1.0 - af);
}

MatrixXf & NeuralNetwork::operator[](int layer)
{
	return Matrices[layer];
}

const RowVectorXf & NeuralNetwork::FeedForward(const vector<float> & input)
{
	std::copy(input.data(), input.data() + input.size(), O[0].data());

	const unsigned int lastIndex = Matrices.size() - 1;
	for (unsigned int m = 1; m < lastIndex; m++)
	{
		Ex[m].noalias() = O[m - 1] * Matrices[m];
		O[m].head(O[m].size() - 1) = Ex[m].unaryExpr(&Activation);
	}

	Ex[lastIndex].noalias() = O[lastIndex - 1] * Matrices[lastIndex];
	O[lastIndex].noalias() = Ex[lastIndex].unaryExpr(&OutActivation);

	return O.back();
}

const RowVectorXf & NeuralNetwork::Evaluate(const vector<float> & input)
{
	std::copy(input.data(), input.data() + input.size(), O[0].data());

	const unsigned int lastIndex = Matrices.size() - 1;
	for (unsigned int m = 1; m < lastIndex; m++)
	{
		O[m].head(O[m].size() - 1).noalias() = O[m - 1] * Matrices[m];
		O[m].head(O[m].size() - 1) = O[m].head(O[m].size() - 1).unaryExpr(&Activation);
	}

	O[lastIndex].noalias() = O[lastIndex - 1] * Matrices[lastIndex];
	O[lastIndex] = O[lastIndex].unaryExpr(&OutActivation);

	return O.back();
}

float NeuralNetwork::Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	unsigned int correct = 0;
	
	for (unsigned int i = 0; i < inputs.size(); i++)
	{
		Evaluate(inputs[i]);
		bool allCorrect = true;
		for (unsigned int r = 0; r < O.back().size(); r++)
		{
			//if (std::signbit(O.back()[r]) != std::signbit(targets[i][r]))
			if (abs(targets[i][r] - O.back()[r]) > 0.5)
			{
				allCorrect = false;
				break;
			}
		}

		if (allCorrect)
			correct++;
	}

	return correct * 100.0 / inputs.size();
}

float NeuralNetwork::SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff)
{
	const unsigned int inputCount = inputs.size();
	float ret = 0;
	for (unsigned int i = 0; i < inputCount; i++)
	{
		Evaluate(inputs[i]);

		const unsigned int resSize = O.back().size();

		float error = 0;
		for (unsigned int k = 0; k < resSize; k++)
			error += (targets[i][k] - O.back()[k]) * (targets[i][k] - O.back()[k]);

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

void NeuralNetwork::SetMatrices(const vector<MatrixXf> & m)
{
	Matrices = m;
}

bool NeuralNetwork::WriteWeightsToFile(const char * path) const
{
	std::ofstream file(path, std::ios::out | std::ios::binary);
	if (!file.is_open())
		return false;

	const unsigned int matricesCount = Matrices.size();
	for (unsigned int m = 1; m < matricesCount; m++)
	{
		const unsigned int curSize = Matrices[m].size() * sizeof(float);
		file.write((const char*)Matrices[m].data(), curSize);
	}

	return true;
}

bool NeuralNetwork::LoadWeightsFromFile(const char * path)
{
	std::ifstream file(path, std::ios::in | std::ios::binary);
	if (!file.is_open())
		return false;

	const unsigned int matricesCount = Matrices.size();
	for (unsigned int m = 1; m < matricesCount; m++)
	{
		const unsigned int curSize = Matrices[m].size() * sizeof(float);
		file.read((char*)Matrices[m].data(), curSize);
	}

	return true;
}

bool NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate, unsigned int batchSize)
{
	if (batchSize == 0 || batchSize > inputs.size())
		batchSize = inputs.size();

	//resize if needed
	if (Index.size() != inputs.size())
	{
		Index.resize(inputs.size());
		for (unsigned int i = 0; i < Index.size(); i++)
			Index[i] = i;
	}

	//shuffle index vector if needed
	if (batchSize != inputs.size())
	{
		for (unsigned int i = Index.size() - 1; i > 0; i--)
		{
			rnd.SetParams(0, i);
			std::swap(Index[i], Index[rnd()]);
		}
	}
	
	const unsigned int L = D.size() - 1;
	unsigned int start = 0;
	unsigned int end = 0;

	//calculate grad for each batch
	while(end < inputs.size())
	{
		start = end;
		end = start + batchSize;
		if (end > inputs.size())
			end = inputs.size();

		for (unsigned int i = start; i < end; i++)
		{
			FeedForward(inputs[Index[i]]);

			//delta for output
			for (unsigned int j = 0; j < O[L].size(); j++)
				D[L][j] = 2 * OutDerivative(Ex[L][j]) * (O[L][j] - targets[Index[i]][j]);

			//delta for hidden
			for (unsigned int l = L; l > 1; l--)
			{
				//calc the sum
				D[l - 1].noalias() = Matrices[l].topRows(D[l - 1].size()) * D[l].transpose();

				//multiply with derivative
				D[l - 1].array() *= Ex[l - 1].unaryExpr(&Derivative).array();
			}

			//calc Grad
			for (unsigned int l = 1; l < Grad.size(); l++)
				Grad[l].noalias() += O[l - 1].transpose() * D[l];
		}

		//update weights
		for (unsigned int l = 1; l < Matrices.size(); l++)
		{
			if (!Grad[l].allFinite())
				return false;

			Matrices[l] -= rate * Grad[l];
		}

		std::swap(Grad, PrevGrad);
		for (unsigned int l = 1; l < Matrices.size(); l++)
			Grad[l].setZero();
	}

	return true;
}

bool NeuralNetwork::TrainRprop(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const unsigned int epochs)
{
	const unsigned int L = D.size() - 1;
	for (unsigned int e = 0; e < epochs; e++)
	{
		for (unsigned int i = 0; i < inputs.size(); i++)
		{
			FeedForward(inputs[i]);

			//delta for output
			for (unsigned int j = 0; j < O[L].size(); j++)
				D[L][j] = 2 * OutDerivative(Ex[L][j]) * (O[L][j] - targets[i][j]);

			//delta for hidden
			for (unsigned int l = L; l > 1; l--)
			{
				//calc the sum
				D[l - 1].noalias() = Matrices[l].topRows(D[l - 1].size()) * D[l].transpose();

				//multiply with derivative
				D[l - 1].array() *= Ex[l - 1].unaryExpr(&Derivative).array();
			}

			//calc Grad
			for (unsigned int l = 1; l < Grad.size(); l++)
				Grad[l].noalias() += O[l - 1].transpose() * D[l];
		}

		//update weights
		for (unsigned int l = 1; l < Matrices.size(); l++)
		{
			if (!Grad[l].allFinite())
				return false;

			//store number of weights here to avoid recalculation of rows * cols for each matrix
			const unsigned int numOfWeights = Matrices[l].size();
			for (unsigned int w = 0; w < numOfWeights; w++)
			{
				const int sign = Sign(Grad[l](w) * PrevGrad[l](w));
				if (sign > 0)
					Delta[l](w) = fmin(Delta[l](w) * 1.2, 100.0);
				else if (sign < 0)
					Delta[l](w) = fmax(Delta[l](w) * 0.5, 1e-10);

				Matrices[l](w) -= Sign(Grad[l](w)) * Delta[l](w);
			}
		}

		std::swap(Grad, PrevGrad);
		for (unsigned int l = 1; l < Matrices.size(); l++)
			Grad[l].setZero();
	}

	return true;
}
