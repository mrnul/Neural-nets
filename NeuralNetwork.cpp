#include <MyHeaders\NeuralNetwork.h>



NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::NeuralNetwork(const vector<unsigned int> topology, const double min, const double max)
{
	Initialize(topology, min, max);
}

void NeuralNetwork::Initialize(const vector<unsigned int> topology, const double min, const double max)
{
	const unsigned int numOfLayers = topology.size();
	const unsigned int lastIndex = numOfLayers - 1;

	Net.resize(numOfLayers);
	D.resize(numOfLayers);
	O.resize(numOfLayers);
	Matrices.resize(numOfLayers);
	DMatrices.resize(numOfLayers);

	for (unsigned int i = 0; i < numOfLayers; i++)
	{
		//+1 for the bias
		//except for the last layer
		if (i != lastIndex)
		{
			O[i].resize(topology[i] + 1);
			O[i].back() = 1;
			//from now on, O[i].back() is always 1
		}
		else
			O[i].resize(topology[i]);
	}

	for (unsigned int i = 1; i < numOfLayers; i++)
	{
		//these two don't need a bias node
		D[i].resize(topology[i]);
		Net[i].resize(topology[i]);

		//+1 for the bias of the prev layer
		//initialize Matrices with random numbers
		Matrices[i].Initialize(topology[i - 1] + 1, topology[i], min, max);

		//initialize DMatrices with value 0
		DMatrices[i].Initialize(topology[i - 1] + 1, topology[i]);
	}

}

double NeuralNetwork::ActivationFunction(const double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::Derivative(const double x)
{
	return x * (1.0 - x);
}

const Matrix & NeuralNetwork::GetMatrix(const unsigned int layer) const
{
	return Matrices[layer];
}

const vector<Matrix> & NeuralNetwork::GetMatrices() const
{
	return Matrices;
}

Matrix & NeuralNetwork::operator[](unsigned int layer)
{
	return Matrices[layer];
}

const vector<double> & NeuralNetwork::Evaluate(vector<double> input)
{
	const unsigned int mCount = Matrices.size();
	unsigned int m = 1;

	//bias node goes last
	input.push_back(1);

	while(true)
	{
		Result.resize(Matrices[m].OutputCount());
		VectorMatrixProduct(input, Matrices[m], Result);

		if (++m == mCount)
			break;

		input.resize(Result.size() + 1);
		
		for (unsigned int i = 0; i < Result.size(); i++)
			input[i] = ActivationFunction(Result[i]);

		input.back() = 1;
	}

	for (unsigned int i = 0; i < Result.size(); i++)
		Result[i] = ActivationFunction(Result[i]);

	return Result;
}

void NeuralNetwork::Evaluate(vector<double> input, vector<double> & output) const
{
	const unsigned int mCount = Matrices.size();
	unsigned int m = 1;

	//bias node goes last
	input.push_back(1);

	while (true)
	{
		output.resize(Matrices[m].OutputCount());
		VectorMatrixProduct(input, Matrices[m], output);

		if (++m == mCount)
			break;

		input.resize(output.size() + 1);
		
		for (unsigned int i = 0; i < output.size(); i++)
			input[i] = ActivationFunction(output[i]);

		input.back() = 1;
	}

	for (unsigned int i = 0; i < output.size(); i++)
		output[i] = ActivationFunction(output[i]);
}

const vector<double> & NeuralNetwork::FeedForward(const vector<double> & input)
{
	for (unsigned int i = 0; i < input.size(); i++)
		O[0][i] = input[i];

	const unsigned int lastIndex = Matrices.size() - 1;
	unsigned int m = 1;
	while (true)
	{
		VectorMatrixProduct(O[m - 1], Matrices[m], Net[m]);

		if (m == lastIndex)
			break;

		for (unsigned int i = 0; i < Net[m].size(); i++)
			O[m][i] = ActivationFunction(Net[m][i]);

		m++;
	}

	for (unsigned int i = 0; i < Net.back().size(); i++)
		O.back()[i] = ActivationFunction(Net.back()[i]);

	return O.back();
}

double NeuralNetwork::Accuracy(const vector<vector<double>> & inputs, const vector<vector<double>> & targets)
{
	unsigned int correct = 0;
	
	for (unsigned int i = 0; i < inputs.size(); i++)
	{
		Evaluate(inputs[i]);
		bool allCorrect = true;
		for (unsigned int r = 0; r < Result.size(); r++)
		{
			if (abs(targets[i][r] - Result[r]) >= 0.5)
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

double NeuralNetwork::Error(const vector<vector<double>> &inputs, const vector<vector<double>> & targets)
{
	const unsigned int inputCount = inputs.size();
	double ret = 0;
	for (unsigned int i = 0; i < inputCount; i++)
	{
		Evaluate(inputs[i]);

		const unsigned int resSize = Result.size();

		double error = 0;
		for (unsigned int k = 0; k < resSize; k++)
			error += (targets[i][k] - Result[k]) * (targets[i][k] - Result[k]);

		ret += error;
	}

	return ret / inputs.size();
}

void NeuralNetwork::SetMatrices(const vector<Matrix> & m)
{
	Matrices = m;
}

unsigned int NeuralNetwork::WriteWeightsToFile(const TCHAR * path) const
{
	File out;
	if (!out.Open(path, file::access::Write, file::share::Read, file::openmode::CreateAlways, file::flag::SeqAccess))
		return 0;

	unsigned int totSize = 0;
	const unsigned int matricesCount = Matrices.size();
	for (unsigned int m = 1; m < matricesCount; m++)
	{
		const unsigned int curSize = Matrices[m].WeightsCount() * sizeof(double);
		if (out.Write(Matrices[m].Data(), curSize) != curSize)
			return 0;

		totSize += curSize;
	}

	return totSize;
}

unsigned int NeuralNetwork::LoadWeightsFromFile(const TCHAR * path)
{
	File in;
	if (!in.Open(path, file::access::Read, file::share::Read, file::openmode::OpenExisting, file::flag::SeqAccess))
		return 0;

	const unsigned int NumOfWeightsInFile = (unsigned int)in.GetSize() / sizeof(double);

	unsigned int totWeights = 0;
	const unsigned int matricesCount = Matrices.size();
	for (unsigned int m = 1; m < matricesCount; m++)
		totWeights += Matrices[m].WeightsCount();

	if (NumOfWeightsInFile != totWeights)
		return 0;

	vector<double> fileData(NumOfWeightsInFile);

	if (in.Read(&fileData[0], fileData.size() * sizeof(double)) != fileData.size() * sizeof(double))
		return 0;

	unsigned int pos = 0;
	for (unsigned int m = 1; m < matricesCount; m++)
	{
		const unsigned int weightsCount = Matrices[m].WeightsCount();
		for (unsigned int w = 0; w < weightsCount; w++)
		{
			Matrices[m].Set(w, fileData[pos++]);
		}
	}

	return pos * sizeof(double);
}

void NeuralNetwork::Train(const vector<vector<double>> & inputs, const vector<vector<double>>& targets, const double rate, const unsigned int parts)
{
	//resize if needed
	if (Index.size() != inputs.size())
	{
		Index.resize(inputs.size());
		for (unsigned int i = 0; i < Index.size(); i++)
			Index[i] = i;
	}

	//shuffle index vector if needed
	if (parts != 1)
	{
		for (unsigned int i = Index.size() - 1; i > 0; i--)
		{
			UniformIntRandom rnd(0, i);
			std::swap(Index[i], Index[rnd()]);
		}
	}
	
	const unsigned int L = D.size() - 1;

	const unsigned batchSize = inputs.size() / parts;

	//split input in equal parts
	//calculate grad for each
	for (unsigned int p = 0; p < parts; p++)
	{
		const unsigned int start = p * batchSize;
		//if input size is not a multiple of 'parts', in the last run take into account those extra examples
		const unsigned int end = start + batchSize + ((p == parts - 1) ? (inputs.size() % batchSize) : 0);

		for (unsigned int i = start; i < end; i++)
		{
			FeedForward(inputs[Index[i]]);

			//delta for output
			//checked only for one output neuron
			//probably won't work for multiple outputs
			for (unsigned int j = 0; j < O[L].size(); j++)
				D[L][j] = 2 * Derivative(O[L][j]) * (O[L][j] - targets[Index[i]][j]);

			//delta for hidden
			for (unsigned int l = L; l > 0; l--)
			{
				for (unsigned int j = 0; j < D[l - 1].size(); j++)
				{
					double sum = 0;
					for (unsigned int q = 0; q < D[l].size(); q++)
						sum += Matrices[l].Element(j, q) * D[l][q];

					D[l - 1][j] = 2 * Derivative(O[l - 1][j]) * sum;
				}
			}

			//calc DMatrices
			for (unsigned int l = 1; l < DMatrices.size(); l++)
			{
				for (unsigned int r = 0; r < DMatrices[l].InputCount(); r++)
				{
					for (unsigned int c = 0; c < DMatrices[l].OutputCount(); c++)
					{
						const double tmp = DMatrices[l].Element(r, c) + rate * O[l - 1][r] * D[l][c];
						DMatrices[l].Set(r, c, tmp);
					}
				}
			}
		}

		//update weights
		for (unsigned int l = 1; l < Matrices.size(); l++)
		{
			//store number of weights here to avoid recalculation of InCount * OutCount for each matrix
			const unsigned int W = Matrices[l].WeightsCount();
			for (unsigned int w = 0; w < W; w++)
			{
				const double tmp = Matrices[l].Element(w) - DMatrices[l].Element(w);
				Matrices[l].Set(w, tmp);
				DMatrices[l].Set(w, 0);
			}
		}
	}
}
