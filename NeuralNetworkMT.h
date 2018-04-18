#pragma once

#include <MyHeaders\NeuralNetwork.h>
#include <thread>
using std::thread;

struct ThreadData
{
	NeuralNetwork nn;
	const vector<vector<float>> *inputs;
	const vector<vector<float>> *targets;

	float l1;
	float l2;
	int Start;
	int End;

	Event wakeUp;
	Event mustQuit;
	Event jobDone;
};

void NeuralNetworkTrainThread(ThreadData & data, NeuralNetwork & master);

class NeuralNetworkMT
{
	private:
		vector<thread> Threads;
		vector<ThreadData> Data;
		NeuralNetwork Master;
		vector<int> Topology;
	public:
		NeuralNetworkMT();
		NeuralNetworkMT(const vector<int> topology, const int threads = thread::hardware_concurrency());
		void Initialize(const vector<int> topology, const int threads = thread::hardware_concurrency());
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		NeuralNetwork & GetMaster();
		void BeginThreads(const int threads);
		void StopThreads();
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate,
			int batchSize = 0, const float l1 = 0.0f, const float l2 = 0.0f);
		void TrainRPROP(const vector<vector<float>> &inputs, const vector<vector<float>> & targets);
};