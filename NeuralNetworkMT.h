#pragma once

#include <MyHeaders\NeuralNetwork.h>
#include <thread>
using std::thread;

struct ThreadData
{
	NeuralNetwork nn;
	const vector<vector<float>> *inputs;
	const vector<vector<float>> *targets;

	float rate;
	int Start;
	int End;

	Event wakeUp;
	Event mustQuit;
	Event jobDone;
};

void NeuralNetworkThread(ThreadData & data, const NeuralNetwork & master);

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
		void BeginThreads(const int threads);
		void StopThreads();
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate, int batchSize = 0);
};