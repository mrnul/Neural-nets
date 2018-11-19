#include <MyHeaders\NeuralNetwork.h>

NeuralNetwork::NeuralNetwork()
{
	std::srand((unsigned int)std::time(0));
}

NeuralNetwork::NeuralNetwork(const vector<int> topology, const int threadCount)
{
	std::srand((unsigned int)std::time(0));
	Initialize(topology , threadCount);
}

void NeuralNetwork::Initialize(const vector<int> topology, const int threadCount)
{
	Base.InitializeBase(topology);
	BeginThreads(threadCount);
}

void NeuralNetwork::BeginThreads(const int threadCount)
{
	Threads.resize(threadCount);
	ThreadData.resize(threadCount);

	auto ThreadFunction = [&](TData & ThreadData)
	{
		//Make sure they have the same dimentions
		ThreadData.Network.Ex = Base.Ex;
		ThreadData.Network.O = Base.O;
		ThreadData.Network.D = Base.D;
		ThreadData.Network.Grad = Base.Grad;
		ThreadData.Network.PrevGrad = Base.PrevGrad;
		
		while (true)
		{
			{
				std::unique_lock<mutex> lock(ThreadData.mtWakeUp);
				ThreadData.cvWakeUp.wait(lock, [&]() {return ThreadData.WakeUp; });
			}

			if (ThreadData.MustQuit)
				break;

			ThreadData.Network.SwapAndZeroGrad();

			for (int i = ThreadData.Start; i < ThreadData.End; i++)
			{
				neuralnetworkbase::RawFeedforward(Base.Matrices, ThreadData.Network.Ex, ThreadData.Network.O, (*Inputs)[Base.Index[i]], Params);
				neuralnetworkbase::RawBackprop(Base.Matrices, ThreadData.Network.Ex, ThreadData.Network.O, ThreadData.Network.D, ThreadData.Network.Grad, (*Targets)[Base.Index[i]]);
			}

			neuralnetworkbase::RawAddL1L2(Base.Matrices, ThreadData.Network.Grad, Params);
			neuralnetworkbase::RawAddMomentum(ThreadData.Network.Grad, ThreadData.Network.PrevGrad, Params);


			{
				std::lock_guard<mutex> lock(ThreadData.mtWakeUp);
				ThreadData.WakeUp = false;
			}
			ThreadData.cvWakeUp.notify_one();
		}

	};

	for (int i = 0; i < threadCount; i++)
		Threads[i] = thread(ThreadFunction, std::ref(ThreadData[i]));
}

void NeuralNetwork::StopThreads()
{
	const int threadCount = (int)Threads.size();
	for (int t = 0; t < threadCount; t++)
	{
		{
			std::lock_guard<mutex> lock(ThreadData[t].mtWakeUp);
			ThreadData[t].WakeUp = true;
			ThreadData[t].MustQuit = true;
		}
		ThreadData[t].cvWakeUp.notify_one();
		Threads[t].join();
	}

	Threads = vector<thread>();
	ThreadData = deque<TData>();
}

const MatrixXf & NeuralNetwork::Evaluate(const vector<float> & input)
{
	//evaluate is feedforward with default parameters
	return Base.Feedforward(input, neuralnetworkbase::NNParams());
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
	Inputs = &inputs;
	Targets = &targets;

	const int inputSize = (int)inputs.size();

	//resize if needed
	if (Base.Index.size() != inputSize)
		Base.InitializeIndexVector(inputSize);

	//shuffle index vector if needed
	if (Params.BatchSize != inputSize)
		Base.ShuffleIndexVector();

	const auto threadsCount = Threads.size();
	const int howManyPerThread = (int)(Params.BatchSize / threadsCount);

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + Params.BatchSize, inputSize);

		for (int t = 0; t < threadsCount; t++)
		{
			ThreadData[t].Start = start + t * howManyPerThread;
			ThreadData[t].End = std::min(ThreadData[t].Start + howManyPerThread, inputSize);

			{
				std::lock_guard<mutex> lock(ThreadData[t].mtWakeUp);
				ThreadData[t].WakeUp = true;
			}
			ThreadData[t].cvWakeUp.notify_one();
		}

		

		Base.SwapAndZeroGrad();
		const auto L = Base.Grad.size();

		//wait for threads to finish and update the weights of Base
		for (int t = 0; t < threadsCount; t++)
		{
			std::unique_lock<mutex> lock(ThreadData[t].mtWakeUp);
			ThreadData[t].cvWakeUp.wait(lock, [&]() {return !ThreadData[t].WakeUp; });

			//now Grad is the Gradient + regularization terms + momentum
			for (int i = 0; i < L; i++)
				Base.Grad[i] += ThreadData[t].Network.Grad[i];

		}

		Base.UpdateWeights(Params);
	}
}

NeuralNetwork::~NeuralNetwork()
{
	StopThreads();
}