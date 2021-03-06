#pragma once

#include <MyHeaders\NeuralNetwork.h>
#include <MyHeaders\Event.h>
#include <thread>
using std::thread;

#undef min //i can't use std::min

struct ThreadData
{
	NeuralNetwork NN;

	Event wakeUp;
	Event jobDone;

	int Start;
	int End;

	bool mustQuit;
};

class NeuralNetworkMT
{
	private:
		vector<thread> Threads;
		vector<ThreadData> Data;
		vector<unsigned int> Topology;
		const vector<vector<float>> *Inputs;
		const vector<vector<float>> *Targets;
	public:
		NeuralNetwork Master;

		NeuralNetworkMT();
		NeuralNetworkMT(const vector<unsigned int> topology, const int threads = thread::hardware_concurrency());
		void Initialize(const vector<unsigned int> topology, const int threads = thread::hardware_concurrency());
		void BeginThreads(const int threads = thread::hardware_concurrency());
		void StopThreads();
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		~NeuralNetworkMT();
};