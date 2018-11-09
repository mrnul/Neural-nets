#include <MyHeaders\NeuralNetwork.h>

NeuralNetwork::NeuralNetwork()
{
	std::srand((unsigned int)std::time(0));
}

NeuralNetwork::NeuralNetwork(const vector<int> topology, const int ThreadCount)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology , ThreadCount);
}

void NeuralNetwork::Initialize(const vector<int> topology, const int ThreadCount)
{
	Base.InitializeBase(topology, ThreadCount);
}

const MatrixXf & NeuralNetwork::Evaluate(const vector<float> & input)
{
	return Base.FeedForward(input, 0.f);
}

float NeuralNetwork::SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff)
{
	const auto inputCount = inputs.size();
	float ret = 0;
	for (int i = 0; i < inputCount; i++)
	{
		Evaluate(inputs[i]);

		const auto resSize = Base.O.back().size();

		float error = 0;
		for (int k = 0; k < resSize; k++)
			error += (targets[i][k] - Base.O.back()(k)) * (targets[i][k] - Base.O.back()(k));

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

float NeuralNetwork::Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	int correct = 0;
	const auto inputSize = inputs.size();
	for (int i = 0; i < inputSize; i++)
	{
		Evaluate(inputs[i]);

		const int oMax = neuralnetworkbase::IndexOfMax(Base.O.back());
		const int tMax = neuralnetworkbase::IndexOfMax(targets[i]);

		if (tMax == oMax)
			correct++;
	}

	return correct * 100.0f / inputSize;
}

bool NeuralNetwork::WriteWeightsToFile(const char * path) const
{
	return Base.WriteWeightsToFile( path);
}

bool NeuralNetwork::LoadWeightsFromFile(const char * path)
{
	return Base.LoadWeightsFromFile(path);
}

void NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	const int inputSize = (int)inputs.size();

	if (inputSize != Base.Index.size())
		Base.InitializeIndexVector(inputSize);

	if (inputSize != Params.BatchSize)
		Base.ShuffleIndexVector();

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + Params.BatchSize, inputSize);

		for (int i = start; i < end; i++)
		{
			Base.FeedForward(inputs[Base.Index[i]], Params.DropOutRate);
			Base.Backprop(targets[Base.Index[i]]);
		}

		Base.AddL1L2(Params.L1, Params.L2);
		Base.AddMomentum(Params.Momentum);
		Base.UpdateWeights(Params.LearningRate, Params.NormalizeGradient);

		Base.SwapAndZeroGrad();
	}
}
