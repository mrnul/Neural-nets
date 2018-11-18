#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
using std::mutex;
using std::thread;
using std::deque;
using std::condition_variable;

struct TData
{
	neuralnetworkbase::NNBase Network;

	int Start;
	int End;

	mutex mtJobDone;
	mutex mtWakeUp;
	condition_variable cvJobDone;
	condition_variable cvWakeUp;

	bool JobDone;
	bool WakeUp;
	bool MustQuit;

	TData() :JobDone(false), WakeUp(false), MustQuit(false)
	{}
};



class NeuralNetwork
{
	private:
		neuralnetworkbase::NNBase Base;
		vector<thread> Threads;
		deque<TData> ThreadData;

		const vector<vector<float>> *Inputs;
		const vector<vector<float>> *Targets;

	public:

		neuralnetworkbase::NNParams Params;

		NeuralNetwork();
		NeuralNetwork(const vector<int> topology, const int threadCount = 1);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<int> topology, const int threadCount = 1);

		void BeginThreads(const int threadCount = 1);
		void StopThreads();

		//returns Base.O.back()
		const MatrixXf & Evaluate(const vector<float> & input);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> & inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);

		bool WriteWeightsToFile(const char * path) const;
		bool LoadWeightsFromFile(const char * path);

		//training with backpropagation, using Params member
		void Train(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		~NeuralNetwork();
};
