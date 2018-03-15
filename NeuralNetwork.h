#pragma once

#include <MyHeaders\Matrix.h>
#include <MyHeaders\File.h>

class NeuralNetwork
{
	private:
		//each layer's matrix
		vector<Matrix> Matrices;

		//each neuron's excitation
		vector<vector<double>> Net;

		//each neuron's output
		vector<vector<double>> O;

		//each neuron's delta
		vector<vector<double>> D;

		vector<Matrix> DMatrices;
		vector<unsigned int> Index;

		vector<double> Result;
	public:
		NeuralNetwork();
		NeuralNetwork(const vector<unsigned int> topology, const double min = 0, const double max = 0);
		static double ActivationFunction(const double x);
		static double Derivative(const double x);
		void Initialize(const vector<unsigned int> topology, const double min = 0, const double max = 0);
		const Matrix & GetMatrix(const unsigned int layer) const;
		const vector<Matrix> & GetMatrices() const;
		Matrix & operator[](unsigned int layer);
		const vector<double> & Evaluate(vector<double> input);
		void Evaluate(vector<double> input, vector<double> & output) const;
		const vector<double> & FeedForward(const vector<double> & input);
		double Accuracy(const vector<vector<double>> & inputs, const vector<vector<double>> & targets);
		double Error(const vector<vector<double>> &inputs, const vector<vector<double>> & targets);
		void SetMatrices(const vector<Matrix> & m);
		unsigned int WriteWeightsToFile(const TCHAR	*path) const;
		unsigned int LoadWeightsFromFile(const TCHAR *path);

		void Train(const vector<vector<double>> &inputs, const vector<vector<double>> & targets, const double rate, const unsigned int parts = 1);
};