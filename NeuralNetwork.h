#pragma once

#include <MyHeaders\XRandom.h>
#include <Eigen\Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
using std::vector;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;


void NormalizeVector(vector<float> & vec, const float a = 0, const float b = 1);
int Sign(const float x);

class NeuralNetwork
{
	private:
		//each layer's matrix
		vector<MatrixXf> Matrices;

		//each layer's excitation
		vector<RowVectorXf> Ex;

		//each neuron's output
		vector<RowVectorXf> O;

		//each neuron's delta
		vector<RowVectorXf> D;

		//to shuffle the index vector
		UniformIntRandom rnd;

		vector<MatrixXf> Grad;
		//previous grad for rprop
		vector<MatrixXf> PrevGrad;
		//delta for rprop
		vector<MatrixXf> Delta;
		//to shuffle inputs
		vector<unsigned int> Index;
	public:
		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology);
		static float Activation(const float x);
		static float Derivative(const float x);
		static float OutActivation(const float x);
		static float OutDerivative(const float x);
		void Initialize(const vector<unsigned int> topology);
		MatrixXf & operator[](int layer);
		const RowVectorXf & FeedForward(const vector<float> & input);
		const RowVectorXf & Evaluate(const vector<float> & input);
		float Accuracy(const vector<vector<float>> & inputs, const vector<vector<float>> & targets);
		float SquareError(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float CutOff = INFINITY);
		void SetMatrices(const vector<MatrixXf> & m);
		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		bool Train(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const float rate, unsigned int batchSize = 0);
		bool TrainRprop(const vector<vector<float>> &inputs, const vector<vector<float>> & targets, const unsigned int epochs = 1);
};
