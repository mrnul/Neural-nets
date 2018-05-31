#pragma once

#include <MyHeaders\NeuralNetwork.h>
#include <MyHeaders\Event.h>
#include <shared_mutex>
#include <thread>
using std::thread;

#undef min //i can't use std::min

struct ThreadData
{
	NeuralNetwork NN;

	Event wakeUp;
	Event jobDone;
	bool mustQuit;

	int Start;
	int End;
};

class NeuralNetworkMT
{
	private:
		vector<thread> Threads;
		vector<ThreadData> Data;
		vector<int> Topology;
		const vector<vector<float>> *Inputs;
		const vector<vector<float>> *Targets;
	public:
		NeuralNetwork Master;

		NeuralNetworkMT();
		NeuralNetworkMT(const vector<int> topology, const int threads = thread::hardware_concurrency());
		void Initialize(const vector<int> topology, const int threads = thread::hardware_concurrency());
		void BeginThreads(const int threads = thread::hardware_concurrency());
		void StopThreads();
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		~NeuralNetworkMT();
};