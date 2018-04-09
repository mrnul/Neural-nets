#include <MyHeaders\NeuralNetworkMT.h>

void NeuralNetworkThread(ThreadData & data, const NeuralNetwork & master)
{
	while (true)
	{
		if (!data.wakeUp.Wait())
			return;

		if (data.mustQuit.IsSignaled())
			return;

		data.nn.SetIndex(master.GetIndex());
		data.nn.SetMatrices(master.GetMatrices());

		data.nn.ZeroGrad();
		data.nn.FeedAndBackProp(data.Start, data.End, *data.inputs, *data.targets);

		data.wakeUp.Reset();
		data.jobDone.Signal();
	}
}

NeuralNetworkMT::NeuralNetworkMT()
{

}

NeuralNetworkMT::NeuralNetworkMT(const vector<int> topology, const int threads)
{
	Initialize(topology, threads);
}

void NeuralNetworkMT::Initialize(const vector<int> topology, const int threads)
{
	Topology = topology;
	Master.Initialize(topology);
	BeginThreads(threads);
}

float NeuralNetworkMT::SquareError(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, const float CutOff)
{
	return Master.SquareError(inputs, targets, CutOff);
}

float NeuralNetworkMT::Accuracy(const vector<vector<float>>& inputs, const vector<vector<float>>& targets)
{
	return Master.Accuracy(inputs, targets);
}

void NeuralNetworkMT::BeginThreads(const int threads)
{
	Data.resize(threads);
	Threads.resize(threads);
	for (int i = 0; i < threads; i++)
	{
		Data[i].nn.Initialize(Topology);
		Threads[i] = thread(NeuralNetworkThread, std::ref(Data[i]), std::cref(Master));
	}	
}

void NeuralNetworkMT::StopThreads()
{
	const int threads = Threads.size();
	for (int i = 0; i < threads; i++)
	{
		Data[i].mustQuit.Signal();
		Data[i].wakeUp.Signal();

		Threads[i].join();
	}

	Data = vector<ThreadData>();
	Threads = vector<thread>();
}

void NeuralNetworkMT::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate, int batchSize)
{
	const int inputSize = inputs.size();

	if (batchSize == 0 || batchSize > inputSize)
		batchSize = inputSize;

	//resize if needed
	if (Master.GetIndex().size() != inputSize)
		Master.ResizeIndex(inputSize);

	//shuffle index vector if needed
	if (batchSize != inputSize)
		Master.ShuffleIndex();

	int end = 0;
	const int threadsCount = Threads.size();

	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + batchSize, inputSize);

		const int howManyPerThread = batchSize / threadsCount;
		for (int t = 0; t < threadsCount; t++)
		{
			Data[t].rate = rate;
			Data[t].inputs = &inputs;
			Data[t].targets = &targets;

			Data[t].Start = start + t * howManyPerThread;
			Data[t].End = std::min(Data[t].Start + howManyPerThread, inputSize);

			Data[t].wakeUp.Signal();
		}

		//wait for threads to finish and update the weights of Master
		for (int t = 0; t < threadsCount; t++)
		{
			Data[t].jobDone.Wait();
			Data[t].jobDone.Reset();
			Master.UpdateWeights(Data[t].nn.GetGrad(), rate);
		}

	}
}
