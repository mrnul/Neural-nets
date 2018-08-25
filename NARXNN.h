#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <Windows.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <map>
using std::map;

class NARXNN
{
	private:
		//each layer's matrix
		vector<MatrixXf> Matrices;

		//each neuron's excitation
		vector<MatrixXf> Ex;

		//each neuron's output
		vector<MatrixXf> O;

		//each neuron's delta
		vector<MatrixXf> D;

		//the gradient
		vector<MatrixXf> Grad;
		vector<MatrixXf> PrevGrad;

		int PastCount;
		int FeaturesPerInput;

		vector<float> Input;
		void ShiftAndAddToInput(const MatrixXf & v);
		void PrepareInput(const vector<vector<float>> & inputs, const int i);

	public:

		NNParams Params;

		NARXNN();
		NARXNN(vector<unsigned int> topology, const unsigned int pastcount, const int ThreadCount = 1);

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(vector<unsigned int> topology, const unsigned int pastcount, const int ThreadCount = 1);
		void SwapGradPrevGrad();
		void ZeroGrad();

		//returns O.back()
		const MatrixXf & FeedForward(const vector<float> & input);
		bool SetInput(const int position, const vector<float> & v);
		void ZeroInput();
		void Generate(const int count, map<char, vector<float> > & ctv, map<vector<float>, char> & vtc);

		//stops calculation when error > cutoff
		float SquareError(const vector<vector<float>> & data, const float CutOff = INFINITY);

		//backprop on one target
		void BackProp(const vector<float> & target);
		//Wnew = Wold + rate * Grad
		void UpdateWeights();
		//Grad += L1term + L2term
		void AddL1L2();
		//Grad += PrevGrad * momentum
		void AddMomentum();
		//checks if all weights are finite (no nan or inf)
		bool AllFinite();

		void Train(const vector<vector<float>> & data);
};
