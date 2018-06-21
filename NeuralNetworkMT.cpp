#include <MyHeaders\NeuralNetworkMT.h>

NeuralNetworkMT::NeuralNetworkMT()
{
	//hmmm...
}

NeuralNetworkMT::NeuralNetworkMT(const vector<unsigned int> topology, const int threads)
{
	Initialize(topology, threads);
}

void NeuralNetworkMT::Initialize(const vector<unsigned int> topology, const int threads)
{
	Topology = topology;
	Master.Initialize(topology);
	BeginThreads(threads);
}

void NeuralNetworkMT::BeginThreads(const int threads)
{
	auto workingThreads = [&](ThreadData & Data)
	{
		while (true)
		{
			if (!Data.wakeUp.Wait())
				break;

			if (Data.mustQuit)
				break;

			Data.NN.SwapGradPrevGrad();
			Data.NN.ZeroGrad();

			//calculate the Gradient + l1term + l2term + momentum 
			Data.NN.FeedAndBackProp(*Inputs, *Targets, Master.GetMatrices(), Master.GetIndexVector(), Data.Start, Data.End);
			Data.NN.AddL1L2(Master.Params.L1, Master.Params.L2, Master.GetMatrices());
			Data.NN.AddMomentum(Master.Params.Momentum);

			Data.wakeUp.Reset();
			Data.jobDone.Signal();
		}
	};
	
	Data.resize(threads);
	Threads.resize(threads);
	for (int i = 0; i < threads; i++)
	{
		//each thread will use master's weights
		Data[i].NN.InitializeNoWeights(Topology);
		Threads[i] = thread(workingThreads, std::ref(Data[i]));
	}	
}

void NeuralNetworkMT::StopThreads()
{
	const int threads = Threads.size();
	for (int i = 0; i < threads; i++)
	{
		Data[i].mustQuit = true;
		Data[i].wakeUp.Signal();

		Threads[i].join();
	}

	Data = vector<ThreadData>();
	Threads = vector<thread>();
}

void NeuralNetworkMT::Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets)
{
	Inputs = &inputs;
	Targets = &targets;

	const int inputSize = inputs.size();

	//resize if needed
	if (Master.GetIndexVector().size() != inputSize)
		Master.ResizeIndexVector(inputSize);

	//shuffle index vector if needed
	if (Master.Params.BatchSize != inputSize)
		Master.ShuffleIndexVector();

	
	const int threadsCount = Threads.size();
	const int howManyPerThread = Master.Params.BatchSize / threadsCount;

	int end = 0;
	while (end < inputSize)
	{
		const int start = end;
		end = std::min(end + Master.Params.BatchSize, inputSize);

		for (int t = 0; t < threadsCount; t++)
		{
			Data[t].Start = start + t * howManyPerThread;
			Data[t].End = std::min(Data[t].Start + howManyPerThread, inputSize);

			Data[t].wakeUp.Signal();
		}

		Master.ZeroGrad();
		vector<MatrixXf> & Grad = Master.GetGrad();
		const int L = Grad.size();

		//wait for threads to finish and update the weights of Master
		for (int t = 0; t < threadsCount; t++)
		{
			Data[t].jobDone.Wait();
			Data[t].jobDone.Reset();
			
			//now Grad is the Gradient + regularization terms + momentum term
			for (int i = 0; i < L; i++)
				Grad[i] += Data[t].NN.GetGrad()[i];
		}

		Master.NormalizeGrad();
		Master.UpdateWeights(Master.Params.LearningRate);
	}
}

NeuralNetworkMT::~NeuralNetworkMT()
{
	StopThreads();
}
