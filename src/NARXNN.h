#pragma once

#include <MyHeaders/neuralnetworkbase.h>
#include <string>
using std::string;

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
		vector<__int16> Data;

		void ShiftAndAddToInput(const MatrixXf & v);
		bool PrepareInput(const int i);
		void PrepareTarget(const int i);

	public:

		neuralnetworkbase::NNParams Params;

		NARXNN();
		NARXNN(const vector<string> paths, vector<int> topologyHiddenOnly, const unsigned int pastcount, const int ThreadCount = 1);
		//initialize weights with random numbers ~ U(-r, r) with r = sqrt(12 / (in + out))
		void Initialize(const vector<string> paths, vector<int> topologyHiddenOnly, const unsigned int pastcount, const int ThreadCount = 1);
		int NumOfUniqueElements();
		//clear Data, Enc, Dec
		void Clear();

		bool WriteWeightsToFile(const char * path) const;
		bool LoadWeightsFromFile(const char * path);

		//returns O.back()
		const MatrixXf & Evaluate(const vector<float> & input);
		void Generate(const char * path, vector<unsigned char> & feed, const int count);

		//stops calculation when error > cutoff
		float SquareError(const float CutOff = INFINITY);

		void Train();
};
