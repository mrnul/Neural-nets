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
	neuralnetworkbase::InitializeBase(Base, topology, ThreadCount);
}

const MatrixXf & NeuralNetwork::Evaluate(const vector<float> & input)
{
	return neuralnetworkbase::FeedForward(Base, input, 0.f);
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
	return neuralnetworkbase::WriteWeightsToFile(Base, path);
}

bool NeuralNetwork::LoadWeightsFromFile(const char * path)
{
	return neuralnetworkbase::LoadWeightsFromFile(Base, path);
}

void NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	const int inputSize = (int)inputs.size();

	if (inputSize != Base.Index.size())
		neuralnetworkbase::InitializeIndexVector(Base, inputSize);

	if (inputSize != Params.BatchSize)
		neuralnetworkbase::ShuffleIndexVector(Base);

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + Params.BatchSize, inputSize);

		for (int i = start; i < end; i++)
		{
			neuralnetworkbase::FeedForward(Base, inputs[Base.Index[i]], Params.DropOutRate);
			neuralnetworkbase::BackProp(Base, targets[Base.Index[i]]);
		}

		neuralnetworkbase::AddL1L2(Base, Params.L1, Params.L2);
		neuralnetworkbase::AddMomentum(Base, Params.Momentum);
		neuralnetworkbase::UpdateWeights(Base, Params.LearningRate, Params.NormalizeGradient);

		neuralnetworkbase::ZeroGradAndSwap(Base);
	}
}
