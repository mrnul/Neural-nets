#include <MyHeaders/NARXNN.h>

void NARXNN::ShiftAndAddToInput(const MatrixXf & v)
{
	std::rotate(Input.begin(), Input.end() - FeaturesPerInput, Input.end());

	for (int i = 0; i < FeaturesPerInput; i++)
		Input[i] = v(i);
}

void NARXNN::PrepareInput(const vector<vector<float>> & data, const int i)
{
	for (int k = 0; k <= PastCount; k++)
	{
		if (i - k >= 0)
			std::copy(data[i - k].begin(), data[i - k].end(), Input.begin() + k * FeaturesPerInput);
		else
			std::fill(Input.begin() + k * FeaturesPerInput, Input.begin() + (k + 1) * FeaturesPerInput, 0.f);
	}
}

NARXNN::NARXNN()
{
	std::srand((unsigned int)std::time(0));
}

NARXNN::NARXNN(vector<unsigned int> topology, const unsigned int pastcount, const int ThreadCount)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology, PastCount = pastcount, ThreadCount);
}

void NARXNN::Initialize(vector<unsigned int> topology, const unsigned int pastcount, const int ThreadCount)
{
	const auto numOfLayers = topology.size();
	const auto lastIndex = numOfLayers - 1;
	PastCount = pastcount;
	FeaturesPerInput = topology[0];

	
	topology[0] = topology[0] * (PastCount + 1);
	Input.resize(topology[0]);

#ifdef _OPENMP	
	omp_set_num_threads(ThreadCount);
#endif

	D.resize(numOfLayers);
	Ex.resize(numOfLayers);
	O.resize(numOfLayers);
	Matrices.resize(numOfLayers);
	Grad.resize(numOfLayers);
	PrevGrad.resize(numOfLayers);

	for (int i = 1; i < numOfLayers; i++)
	{
		//these don't need a bias node
		D[i].setZero(1, topology[i]);
		Ex[i].setZero(1, topology[i]);

		//these need a bias node
		O[i - 1].setZero(1, topology[i - 1] + 1);
		O[i - 1](topology[i - 1]) = 1;

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
	O[lastIndex].setZero(1, topology[lastIndex]);
}

void NARXNN::SwapGradPrevGrad()
{
	std::swap(Grad, PrevGrad);
}

void NARXNN::ZeroGrad()
{
	const auto GradSize = Grad.size();
	for (int l = 1; l < GradSize; l++)
		Grad[l].setZero();
}

const MatrixXf & NARXNN::FeedForward(const vector<float> & input)
{
	return NNFeedForward(input, Matrices, Ex, O, 0.f);
}

bool NARXNN::SetInput(const int position, const vector<float > & v)
{
	if (position > PastCount)
		return false;
	
	const int startIndex = position * FeaturesPerInput;
	
	std::copy(v.begin(), v.end(), Input.begin() + startIndex);

	return true;
}

void NARXNN::ZeroInput()
{
	std::fill(Input.begin(), Input.end(), 0.f);
}

void NARXNN::Generate(const int count, map<char, vector<float> > & ctv, map<vector<float>, char> & vtc)
{
	for (int c = 0; c < count; c++)
	{
		//predict next char
		FeedForward(Input);

		//print it
		const int iMax = IndexOfMax(O.back());

		vector<float> v(FeaturesPerInput);
		v[iMax] = 1.f;
		std::cout << vtc[v];

		//add it as next input
		ShiftAndAddToInput(O.back());
	}

	std::cout << std::endl;
}

float NARXNN::SquareError(const vector<vector<float>> & data, const float CutOff)
{
	const auto inputCount = data.size() - 1;
	float ret = 0;
	for (int i = PastCount; i < inputCount; i++)
	{
		PrepareInput(data, i);
		FeedForward(Input);

		const auto resSize = O.back().size();

		float error = 0;
		for (int k = 0; k < resSize; k++)
			error += (data[i + 1][k] - O.back()(k)) * (data[i + 1][k] - O.back()(k));

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

void NARXNN::BackProp(const vector<float> & target)
{
	NNBackProp(target, Matrices, Grad, Ex, O, D);
}

void NARXNN::UpdateWeights()
{
	NNUpdateWeights(Matrices, Grad, Params.LearningRate, Params.NormalizeGradient);
}

void NARXNN::AddL1L2()
{
	NNAddL1L2(Params.L1, Params.L2, Matrices, Grad);
}

void NARXNN::AddMomentum()
{
	NNAddMomentum(Params.Momentum, Grad, PrevGrad);
}

bool NARXNN::AllFinite()
{
	const auto size = Matrices.size();
	for (int m = 0; m < size; m++)
	{
		if (!Matrices[m].allFinite())
			return false;
	}

	return true;
}

void NARXNN::Train(const vector<vector<float>> & data)
{
	const int inputSize = (int)data.size();

	for (int b = 0; b < Params.BatchSize; b++)
	{
		const int i = rand() % (inputSize - 1);

		PrepareInput(data, i);
		FeedForward(Input);

		BackProp(data[i + 1]);
	}

	AddL1L2();
	AddMomentum();
	UpdateWeights();

	SwapGradPrevGrad();
	ZeroGrad();
}
