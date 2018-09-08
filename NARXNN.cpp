#include <MyHeaders/NARXNN.h>

void NARXNN::ShiftAndAddToInput(const MatrixXf & v)
{
	std::rotate(Input.begin(), Input.begin() + FeaturesPerInput, Input.end());

	const int first = PastCount * FeaturesPerInput;
	for (int i = first; i < Input.size(); i++)
		Input[i] = v(i - first);
}

void NARXNN::PrepareInput(const int i)
{
	std::fill(Input.begin(), Input.end(), 0.f);

	for (int k = 0, Index = i - PastCount; k <= PastCount; k++, Index++)
	{
		if (Index >= 0)
			Input[k * FeaturesPerInput + Enc[Data[Index]]] = 1.f;
	}
}

void NARXNN::PrepareTarget(const int i)
{
	std::fill(Target.begin(), Target.end(), 0.f);
	Target[Enc[Data[i]]] = 1.f;
}

NARXNN::NARXNN()
{
	std::srand((unsigned int)std::time(0));
}

void NARXNN::ProcessData(const char * path)
{
	neuralnetworkbase::OneHotEncDec(path, Enc, Dec, Data);
}

int NARXNN::NumUniqueElements()
{
	return (int)Enc.size();
}

void NARXNN::ClearData()
{
	Data.clear();
	Enc.clear();
	Dec.clear();
}

void NARXNN::Initialize(vector<int> topology, const unsigned int pastcount, const int ThreadCount)
{
	PastCount = pastcount;
	FeaturesPerInput = topology[0];
	Target.resize(FeaturesPerInput);

	topology[0] = topology[0] * (PastCount + 1);
	Input.resize(topology[0]);

	Base.InitializeBase(topology, ThreadCount);
}

bool NARXNN::WriteWeightsToFile(const char * path) const
{
	return Base.WriteWeightsToFile(path);
}

bool NARXNN::LoadWeightsFromFile(const char * path)
{
	return Base.LoadWeightsFromFile(path);
}

const MatrixXf & NARXNN::Evaluate(const vector<float> & input)
{
	return Base.FeedForward(input, 0.f);
}

void NARXNN::Generate(const char * path, vector<unsigned char> & feed, const int count)
{
	std::ofstream ofile(path, std::ios::binary | std::ios::out);

	std::fill(Input.begin(), Input.end(), 0.f);

	const int size = (int)feed.size() < PastCount + 1 ? (int)feed.size() : PastCount + 1;
	for (int i = 0, index = (int)feed.size() - size; i < size; i++, index++)
		Input[i * FeaturesPerInput + Enc[feed[index]]] = 1.f;

	ofile.write((char*)&feed[0], feed.size());

	for (int c = 0; c < count; c++)
	{
		//predict next char
		Evaluate(Input);

		const int iMax = neuralnetworkbase::IndexOfMax(Base.O.back());
		const unsigned char tmp = Dec[iMax];
		ofile.write((char*)&tmp, 1);

		//Base.O.back().setZero();
		//Base.O.back()(iMax) = 1.f;

		ShiftAndAddToInput(Base.O.back());
	}
}

float NARXNN::SquareError(const float CutOff)
{
	const auto inputCount = Data.size() - 1;
	float ret = 0;
	for (int i = PastCount; i < inputCount; i++)
	{
		PrepareInput(i);
		PrepareTarget(i + 1);

		Evaluate(Input);

		const auto resSize = Base.O.back().size();

		float error = 0;
		for (int k = 0; k < resSize; k++)
			error += (Target[k] - Base.O.back()(k)) * (Target[k] - Base.O.back()(k));

		ret += error;

		if (ret > CutOff)
			break;
	}

	return ret;
}

void NARXNN::Train()
{
	const int inputSize = (int)Data.size() - 1;

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
			PrepareInput(Base.Index[i]);
			PrepareTarget(Base.Index[i] + 1);

			Base.FeedForward(Input, Params.DropOutRate);
			Base.Backprop(Target);
		}

		Base.AddL1L2(Params.L1, Params.L2);
		Base.AddMomentum(Params.Momentum);
		Base.UpdateWeights(Params.LearningRate, Params.NormalizeGradient);

		Base.ZeroGradAndSwap();
	}
}
