#include <MyHeaders\NeuralNetworkMT.h>

void NeuralNetworkTrainThread(ThreadData & data, NeuralNetwork & master)
{
	while (true)
	{
		if (!data.wakeUp.Wait())
			break;

		if (data.mustQuit.IsSignaled())
			break;

		data.nn.SetIndexVector(master.GetIndexVector());
		data.nn.SetMatrices(master.GetMatrices());

		data.nn.SwapGradPrevGrad();
		data.nn.ZeroGrad();
		//calc the grad
		data.nn.FeedAndBackProp(*data.inputs, *data.targets, data.Start, data.End);
		//l1term + l2term
		data.nn.AddL1L2(data.l1, data.l2);
		//add momentum
		data.nn.AddMomentum(data.momentum);

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

NeuralNetwork & NeuralNetworkMT::GetMaster()
{
	return Master;
}

void NeuralNetworkMT::BeginThreads(const int threads)
{
	Data.resize(threads);
	Threads.resize(threads);
	for (int i = 0; i < threads; i++)
	{
		Data[i].nn.Initialize(Topology);
		Threads[i] = thread(NeuralNetworkTrainThread, std::ref(Data[i]), std::ref(Master));
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

void NeuralNetworkMT::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate,
	const float momentum, int batchSize, const float l1, const float l2)
{
	const int inputSize = inputs.size();

	if (batchSize == 0 || batchSize > inputSize)
		batchSize = inputSize;

	//resize if needed
	if (Master.GetIndexVector().size() != inputSize)
		Master.ResizeIndexVector(inputSize);

	//shuffle index vector if needed
	if (batchSize != inputSize)
		Master.ShuffleIndexVector();

	int end = 0;
	const int threadsCount = Threads.size();

	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + batchSize, inputSize);

		const int howManyPerThread = batchSize / threadsCount;
		for (int t = 0; t < threadsCount; t++)
		{
			Data[t].momentum = momentum;
			Data[t].l1 = l1;
			Data[t].l2 = l2;
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

			//Grad has the regularization terms + the momentum
			Master.UpdateWeights(Data[t].nn.GetGrad(), rate);
		}
	}
}
