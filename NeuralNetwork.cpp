#include <MyHeaders\NeuralNetwork.h>

NeuralNetwork::NeuralNetwork()
{
	std::srand((unsigned int)std::time(0));
}

NeuralNetwork::NeuralNetwork(const vector<unsigned int> topology, const int ThreadCount)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology , ThreadCount);
}

void NeuralNetwork::Initialize(const vector<unsigned int> topology, const int ThreadCount)
{
	const auto numOfLayers = topology.size();
	const auto lastIndex = numOfLayers - 1;
 
#ifdef _OPENMP	
	omp_set_num_threads(ThreadCount);
#endif

	Base.D.resize(numOfLayers);
	Base.Ex.resize(numOfLayers);
	Base.O.resize(numOfLayers);
	Base.Matrices.resize(numOfLayers);
	Base.Grad.resize(numOfLayers);
	Base.PrevGrad.resize(numOfLayers);

	for (int i = 1; i < numOfLayers; i++)
	{
		//these don't need a bias node
		Base.D[i].setZero(1, topology[i]);
		Base.Ex[i].setZero(1, topology[i]);

		//these need a bias node
		Base.O[i - 1].setZero(1, topology[i - 1] + 1);
		Base.O[i - 1](topology[i - 1]) = 1;

		//+1 for the bias of the prev layer
		//initialize Matrices with random numbers
		Base.Matrices[i].setRandom(topology[i - 1] + 1, topology[i]);
		Base.Matrices[i].topRows(Base.Matrices[i].rows() - 1) *= sqrt(12.0f / (topology[i - 1] + 1.0f + topology[i]));
		//set biases to zero
		Base.Matrices[i].row(Base.Matrices[i].rows() - 1).setZero();

		//initialize Grad with value 0
		Base.Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		Base.PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);
	}

	//last layer does not have a bias node
	Base.O[lastIndex].setZero(1, topology[lastIndex]);
}

void NeuralNetwork::SwapGradPrevGrad()
{
	std::swap(Base.Grad, Base.PrevGrad);
}

void NeuralNetwork::ZeroGrad()
{
	const auto GradSize = Base.Grad.size();
	#pragma omp parallel for
	for (int l = 1; l < GradSize; l++)
		Base.Grad[l].setZero();
}

void NeuralNetwork::ShuffleIndexVector()
{
	neuralnetworkbase::ShuffleIndexVector(Base);
}

void NeuralNetwork::ResizeIndexVector(const int size)
{
	neuralnetworkbase::InitializeIndexVector(Base, size);
}

const MatrixXf & NeuralNetwork::FeedForward(const vector<float> & input)
{
	return neuralnetworkbase::FeedForward(Base, input, 0.f);
}

float NeuralNetwork::SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff)
{
	const auto inputCount = inputs.size();
	float ret = 0;
	for (int i = 0; i < inputCount; i++)
	{
		FeedForward(inputs[i]);

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
		FeedForward(inputs[i]);

		const int oMax = neuralnetworkbase::IndexOfMax(Base.O.back());
		const int tMax = neuralnetworkbase::IndexOfMax(targets[i]);

		if (tMax == oMax)
			correct++;
	}

	return correct * 100.0f / inputSize;
}

bool NeuralNetwork::WriteWeightsToFile(const char * path) const
{
	std::ofstream file(path, std::ios::out | std::ios::binary);
	if (!file.is_open())
		return false;

	const auto matricesCount = Base.Matrices.size();
	for (int m = 1; m < matricesCount; m++)
	{
		const auto curSize = Base.Matrices[m].size() * sizeof(float);
		file.write((const char*)Base.Matrices[m].data(), curSize);
	}

	return true;
}

bool NeuralNetwork::LoadWeightsFromFile(const char * path)
{
	std::ifstream file(path, std::ios::in | std::ios::binary);
	if (!file.is_open())
		return false;

	const auto matricesCount = Base.Matrices.size();
	for (int m = 1; m < matricesCount; m++)
	{
		const auto curSize = Base.Matrices[m].size() * sizeof(float);
		file.read((char*)Base.Matrices[m].data(), curSize);
	}

	return true;
}

void NeuralNetwork::FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const int start, int end)
{
	neuralnetworkbase::FeedAndBackProp(Base, inputs, targets, start, end, Params.DropOutRate);
}

void NeuralNetwork::BackProp(const vector<float> & target)
{
	neuralnetworkbase::BackProp(Base, target);
}

void NeuralNetwork::UpdateWeights()
{
	neuralnetworkbase::UpdateWeights(Base, Params.LearningRate, Params.NormalizeGradient);
}

void NeuralNetwork::AddL1L2()
{
	neuralnetworkbase::AddL1L2(Base, Params.L1, Params.L2);
}

void NeuralNetwork::AddMomentum()
{
	neuralnetworkbase::AddMomentum(Base, Params.Momentum);
}

void NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	const int inputSize = (int)inputs.size();

	//resize if needed
	if (Base.Index.size() != inputSize)
		ResizeIndexVector(inputSize);

	//shuffle index vector if needed
	if (Params.BatchSize != inputSize)
		ShuffleIndexVector();

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + Params.BatchSize, inputSize);

		FeedAndBackProp(inputs, targets, start, end);
		AddL1L2();
		AddMomentum();
		UpdateWeights();

		SwapGradPrevGrad();
		ZeroGrad();
	}
}
