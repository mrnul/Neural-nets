#include <MyHeaders\NeuralNetwork.h>

void NormalizeVector(vector<double> & vec, const double a, const double b)
{
	double min = vec[0];
	double max = vec[0];

	for (unsigned int i = 1; i < vec.size(); i++)
	{
		if (min > vec[i])
			min = vec[i];
		else if (max < vec[i])
			max = vec[i];
	}

	const double denom = max - min;
	const double coeff = b - a;
	for (unsigned int i = 0; i < vec.size(); i++)
		vec[i] = coeff * (vec[i] - min) / denom + a;
}

int Sign(const double x)
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
	PrevGrad.resize(numOfLayers);
	Delta.resize(numOfLayers);

	//initialize the rest beginning from the second layer
	for (unsigned int i = 1; i < numOfLayers; i++)
	{
		//these don't need a bias node
		D[i].setZero(topology[i]);
		Ex[i].setZero(topology[i]);

		//+1 for the bias of the prev layer
		O[i - 1].setZero(topology[i - 1] + 1);
		O[i - 1][topology[i - 1]] = 1;

		//initialize Matrices with random numbers
		Matrices[i].setRandom(topology[i - 1] + 1, topology[i]);

		//initialize Grads with value 0
		Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);

		//initialize Delta
		Delta[i].setConstant(topology[i - 1] + 1, topology[i], 0.0125);
	}

	O[lastIndex].setZero(topology[lastIndex]);
}

double NeuralNetwork::ActivationFunction(const double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::Derivative(const double x)
{
	const double af = ActivationFunction(x);
	return af * (1.0 - af);
}

MatrixXd & NeuralNetwork::operator[](unsigned int layer)
{
	return Matrices[layer];
}

const RowVectorXd & NeuralNetwork::FeedForward(const vector<double> & input)
{
	for (unsigned int i = 0; i < input.size(); i++)
		O[0][i] = input[i];
	
	const unsigned int lastIndex = Matrices.size() - 1;
	for (unsigned int m = 1; m <= lastIndex; m++)
		O[m].head(O[m].size() - (m != lastIndex)) = (Ex[m].noalias() = O[m - 1] * Matrices[m]).unaryExpr(&ActivationFunction);

	return O.back();
}

double NeuralNetwork::Accuracy(const vector<vector<double>> & inputs, const vector<vector<double>> & targets)
{
	unsigned int correct = 0;
	
	for (unsigned int i = 0; i < inputs.size(); i++)
	{
		FeedForward(inputs[i]);
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

double NeuralNetwork::Error(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const double CutOff)
{
	const unsigned int inputCount = inputs.size();
	double ret = 0;
	for (unsigned int i = 0; i < inputCount; i++)
	{
		FeedForward(inputs[i]);

		const unsigned int resSize = O.back().size();

		double error = 0;
		for (unsigned int k = 0; k < resSize; k++)
			error += (targets[i][k] - O.back()[k]) * (targets[i][k] - O.back()[k]);

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

void NeuralNetwork::SetMatrices(const vector<MatrixXd> & m)
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
		const unsigned int curSize = Matrices[m].size() * sizeof(double);
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
		const unsigned int curSize = Matrices[m].size() * sizeof(double);
		file.read((char*)Matrices[m].data(), curSize);
	}

	return true;
}

void NeuralNetwork::Train(const vector<vector<double>> & inputs, const vector<vector<double>> & targets, const double rate, unsigned int batchSize)
{
	//resize if needed
	if (Index.size() != inputs.size())
	{
		Index.resize(inputs.size());
		for (unsigned int i = 0; i < Index.size(); i++)
			Index[i] = i;
	}

	if (batchSize == 0)
		batchSize = inputs.size();

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

	//split input in equal parts
	//calculate grad for each
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
				D[L][j] = 2 * Derivative(Ex[L][j]) * (O[L][j] - targets[Index[i]][j]);

			//delta for hidden
			for (unsigned int l = L; l > 0; l--)
			{
				for (unsigned int j = 0; j < D[l - 1].size(); j++)
				{
					D[l - 1][j] = Derivative(Ex[l - 1][j]) * Matrices[l].row(j).dot(D[l]);
				}
			}

			//calc Grad
			for (unsigned int l = 1; l < Grad.size(); l++)
			{
				for (unsigned int r = 0; r < Grad[l].rows(); r++)
				{
					Grad[l].row(r).noalias() += rate * (O[l - 1][r] * D[l]);
				}
			}
		}

		//update weights
		for (unsigned int l = 1; l < Matrices.size(); l++)
		{
			Matrices[l] -= Grad[l];
			Grad[l].setZero();
		}

		
	}
}

void NeuralNetwork::TrainRprop(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, const unsigned int epochs)
{
	const unsigned int L = D.size() - 1;
	for (unsigned int e = 0; e < epochs; e++)
	{
		for (unsigned int i = 0; i < inputs.size(); i++)
		{
			FeedForward(inputs[i]);

			//delta for output
			for (unsigned int j = 0; j < O[L].size(); j++)
				D[L][j] = 2 * Derivative(Ex[L][j]) * (O[L][j] - targets[i][j]);

			//delta for hidden
			for (unsigned int l = L; l > 0; l--)
			{
				for (unsigned int j = 0; j < D[l - 1].size(); j++)
				{
					D[l - 1][j] = Derivative(Ex[l - 1][j]) * Matrices[l].row(j).dot(D[l]);
				}
			}

			//calc Grad
			for (unsigned int l = 1; l < Grad.size(); l++)
			{
				for (unsigned int r = 0; r < Grad[l].rows(); r++)
				{
					Grad[l].row(r).noalias() += O[l - 1][r] * D[l];
				}
			}
		}

		//update weights
		for (unsigned int l = 1; l < Matrices.size(); l++)
		{
			//store number of weights here to avoid recalculation of InCount * OutCount for each matrix
			const unsigned int W = Matrices[l].size();
			for (unsigned int w = 0; w < W; w++)
			{
				const int sign = Sign(Grad[l](w) * PrevGrad[l](w));
				if (sign > 0)
					Delta[l](w) = fmin(Delta[l](w) * 1.2, 50.0);
				else if (sign < 0)
					Delta[l](w) = fmax(Delta[l](w) * 0.5, 1e-10);

				Matrices[l](w) -= Sign(Grad[l](w)) * Delta[l](w);
			}
		}

		std::swap(Grad, PrevGrad);
		for (unsigned int l = 1; l < Matrices.size(); l++)
			Grad[l].setZero();
	}
}
