#pragma once

#include <MyHeaders\XRandom.h>
#include <Eigen\Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
using std::vector;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;


void NormalizeVector(vector<double> & vec, const double a = 0, const double b = 1);
int Sign(const double x);

class NeuralNetwork
{
	private:
		//each layer's matrix
		vector<MatrixXd> Matrices;

		//each layer's excitation
		vector<RowVectorXd> Ex;

		//each neuron's output
		vector<RowVectorXd> O;

		//each neuron's delta
		vector<RowVectorXd> D;

		//to shuffle the index vector
		UniformIntRandom rnd;

		vector<MatrixXd> Grad;
		//previous grad for rprop
		vector<MatrixXd> PrevGrad;
		//delta for rprop
		vector<MatrixXd> Delta;
		vector<unsigned int> Index;
	public:
		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology);
		static double ActivationFunction(const double x);
		static double Derivative(const double x);
		void Initialize(const vector<unsigned int> topology);
		MatrixXd & operator[](unsigned int layer);
		const RowVectorXd & FeedForward(const vector<double> & input);
		double Accuracy(const vector<vector<double>> & inputs, const vector<vector<double>> & targets);
		double Error(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const double CutOff = INFINITY);
		void SetMatrices(const vector<MatrixXd> & m);
		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		void Train(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const double rate, unsigned int batchSize = 0);
		void TrainRprop(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const unsigned int epochs = 1);
};
