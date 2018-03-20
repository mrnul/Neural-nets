#pragma once

#include <MyHeaders\Matrix.h>
#include <fstream>

class NeuralNetwork
{
	private:
		//each layer's matrix
		vector<Matrix> Matrices;

		//each neuron's output
		vector<vector<double>> O;

		//each neuron's delta
		vector<vector<double>> D;

		//to shuffle the index vector
		UniformIntRandom rnd;

		vector<Matrix> DMatrices;
		vector<unsigned int> Index;
	public:
		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology, const double min = 0, const double max = 0);
		static double ActivationFunction(const double x);
		static double Derivative(const double x);
		void Initialize(const vector<unsigned int> topology, const double min = 0, const double max = 0);
		const Matrix & GetMatrix(const unsigned int layer) const;
		const vector<Matrix> & GetMatrices() const;
		Matrix & operator[](unsigned int layer);
		const vector<double> & FeedForward(const vector<double> & input);
		double Accuracy(const vector<vector<double>> & inputs, const vector<vector<double>> & targets);
		double Error(const vector<vector<double>> &inputs, const vector<vector<double>> & targets);
		void SetMatrices(const vector<Matrix> & m);
		bool WriteWeightsToFile(const char *path) const;
		bool LoadWeightsFromFile(const char *path);

		void Train(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const double rate, const unsigned int parts = 1);
};