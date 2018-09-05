#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <map>
using std::map;

class NARXNN
{
	private:
		neuralnetworkbase::NNBase Base;

		int PastCount;
		int FeaturesPerInput;

		vector<float> Input;
		vector<float> Target;

		map<unsigned char, int> Enc;
		map<int, unsigned char> Dec;
		vector<unsigned char> Data;

		void ShiftAndAddToInput(const MatrixXf & v);
		void PrepareInput(const int i);
		void PrepareTarget(const int i);

	public:

		neuralnetworkbase::NNParams Params;

		NARXNN();

		//load and insert data from file to Data vector and update Enc - Dec maps
		void ProcessData(const char * path);
		int NumUniqueElements();
		//clear Data, Enc, Dec
		void ClearData();

		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(vector<int> topology, const unsigned int pastcount, const int ThreadCount = 1);

		//returns O.back()
		const MatrixXf & Evaluate(const vector<float> & input);
		void Generate(const char * path, vector<unsigned char> & feed, const int count);

		//stops calculation when error > cutoff
		float SquareError(const float CutOff = INFINITY);

		void Train();
};
