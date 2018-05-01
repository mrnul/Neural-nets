#pragma once

#include <MyHeaders\NeuralNetwork.h>
#include <MyHeaders\Event.h>
#include <thread>
using std::thread;

#undef min //i can't use std::min
#undef max //i can't use std::max

struct ThreadData
{
	NeuralNetwork nn;
	const vector<vector<float>> *inputs;
	const vector<vector<float>> *targets;

	float momentum;
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
		NeuralNetwork & GetMaster();
		void BeginThreads(const int threads);
		void StopThreads();
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float rate,
			const float momentum = 0.0f, int batchSize = 0, const float l1 = 0.0f, const float l2 = 0.0f);
};