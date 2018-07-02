#include <MyHeaders\NeuralNetwork.h>

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
	const int numOfLayers = topology.size();
	const int lastIndex = numOfLayers - 1;

	D.resize(numOfLayers);
	Ex.resize(numOfLayers);
	O.resize(numOfLayers);
	Matrices.resize(numOfLayers);
	Grad.resize(numOfLayers);
	PrevGrad.resize(numOfLayers);

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
		Matrices[i].topRows(Matrices[i].rows() - 1) *= sqrt(12.0f / (topology[i - 1] + 1.0f + topology[i]));
		//set biases to zero
		Matrices[i].row(Matrices[i].rows() - 1).setZero();

		//initialize Grad with value 0
		Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);
	}

	//last layer does not have a bias node
	O[lastIndex].setZero(topology[lastIndex]);
}

void NeuralNetwork::InitializeNoWeights(const vector<unsigned int> topology)
{
	const int numOfLayers = topology.size();
	const int lastIndex = numOfLayers - 1;

	D.resize(numOfLayers);
	Ex.resize(numOfLayers);
	O.resize(numOfLayers);
	Grad.resize(numOfLayers);
	PrevGrad.resize(numOfLayers);

	for (int i = 1; i < numOfLayers; i++)
	{
		//these don't need a bias node
		D[i].setZero(topology[i]);
		Ex[i].setZero(topology[i]);

		//these need a bias node
		O[i - 1].setZero(topology[i - 1] + 1);
		O[i - 1][topology[i - 1]] = 1;

		//initialize Grad with value 0
		Grad[i].setZero(topology[i - 1] + 1, topology[i]);
		PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);
	}

	//last layer does not have a bias node
	O[lastIndex].setZero(topology[lastIndex]);
}

void NeuralNetwork::SetMatrices(const vector<MatrixXf> & m)
{
	Matrices = m;
}

void NeuralNetwork::SetGrad(const vector<MatrixXf> & grad)
{
	Grad = grad;
}

void NeuralNetwork::SetPrevGrad(const vector<MatrixXf> & grad)
{
	PrevGrad = grad;
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

void NeuralNetwork::NormalizeGrad()
{
	const int size = Grad.size();
	float norm = 0;

	for (int i = 0; i < size; i++)
		norm += Grad[i].squaredNorm();

	norm = sqrt(norm);
	for (int i = 0; i < size; i++)
		Grad[i] /= norm;
}

void NeuralNetwork::SwapGradPrevGrad()
{
	std::swap(Grad, PrevGrad);
}

void NeuralNetwork::ZeroGrad()
{
	const int GradSize = Grad.size();
	for (int l = 1; l < GradSize; l++)
		Grad[l].setZero();
}

vector<int> & NeuralNetwork::GetIndexVector()
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
	return NNFeedForward(input, Matrices, Ex, O);
}

const RowVectorXf & NeuralNetwork::FeedForward(const vector<float>& input, const vector<MatrixXf> & matrices)
{
	return NNFeedForward(input, matrices, Ex, O);
}

float NeuralNetwork::SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff)
{
	const int inputCount = inputs.size();
	float ret = 0;
	for (int i = 0; i < inputCount; i++)
	{
		FeedForward(inputs[i]);

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
		FeedForward(inputs[i]);

		const int outSize = O.back().size();
		int tMax = 0;
		int oMax = 0;
		for (int r = 0; r < outSize; r++)
		{
			if (targets[i][tMax] < targets[i][r])
				tMax = r;
			if (O.back()[oMax] < O.back()[r])
				oMax = r;
		}

		if (outSize == 1)
		{
			if ((O.back()[0] > 0.5f && targets[i][0] > 0.5f)
				||
				(O.back()[0] < 0.5f && targets[i][0] < 0.5))
			{
				correct++;
			}
		}
		else if (tMax == oMax)
		{
			correct++;
		}
	}

	return correct * 100.0f / inputSize;
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
	const int start, int end)
{
	NNFeedAndBackProp(inputs, targets, Matrices, Grad, Index, Ex, O, D, start, end);
}

void NeuralNetwork::FeedAndBackProp(const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
	const vector<MatrixXf> & matrices, const vector<int> & index,
	const int start, int end)
{
	NNFeedAndBackProp(inputs, targets, matrices, Grad, index, Ex, O, D, start, end);
}

void NeuralNetwork::BackProp(const vector<float> & target)
{
	NNBackProp(target, Matrices, Grad, Ex, O, D);
}

void NeuralNetwork::BackProp(const vector<float> & target, const vector<MatrixXf> & matrices)
{
	NNBackProp(target, matrices, Grad, Ex, O, D);
}

void NeuralNetwork::UpdateWeights(const vector<MatrixXf> & grad, const float rate)
{
	const int MatricesSize = Matrices.size();
	for (int l = 1; l < MatricesSize; l++)
		Matrices[l] -= rate * grad[l];
}

void NeuralNetwork::UpdateWeights(const float rate)
{
	UpdateWeights(Grad, rate);
}

void NeuralNetwork::AddL1L2(const float l1, const float l2)
{
	AddL1L2(l1, l2, Matrices);
}

void NeuralNetwork::AddL1L2(const float l1, const float l2, const vector<MatrixXf> & matrices)
{
	//Grad = Grad + l1term + l2term
	//don't regularize the bias
	//topRows(Grad[l].rows() - 1) skips the last row (the biases)

	const int lCount = Grad.size();

	//both l1 and l2
	if (l1 != 0.0f && l2 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l].topRows(Grad[l].rows() - 1) +=
			l1 * matrices[l].topRows(Grad[l].rows() - 1).unaryExpr(&Sign)
			+ l2 * matrices[l].topRows(Grad[l].rows() - 1);
	}
	//only l1
	else if (l1 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l].topRows(Grad[l].rows() - 1) += l1 * matrices[l].topRows(Grad[l].rows() - 1).unaryExpr(&Sign);
	}
	//only l2
	else if (l2 != 0.0f)
	{
		for (int l = 1; l < lCount; l++)
			Grad[l].topRows(Grad[l].rows() - 1) += l2 * matrices[l].topRows(Grad[l].rows() - 1);
	}
}

void NeuralNetwork::AddMomentum(const float momentum)
{
	if (momentum == 0.0f)
		return;

	const int lCount = Grad.size();
	for (int l = 0; l < lCount; l++)
		Grad[l] += momentum * PrevGrad[l];
}

bool NeuralNetwork::AllFinite()
{
	const int size = Matrices.size();
	for (int m = 0; m < size; m++)
	{
		if (!Matrices[m].allFinite())
			return false;
	}

	return true;
}

void NeuralNetwork::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	const int inputSize = inputs.size();

	//resize if needed
	if (Index.size() != inputSize)
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
		AddL1L2(Params.L1, Params.L2);
		AddMomentum(Params.Momentum);
		UpdateWeights(Params.LearningRate);

		SwapGradPrevGrad();
		ZeroGrad();
	}
}
