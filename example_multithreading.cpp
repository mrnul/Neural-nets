#include <MyHeaders\NeuralNetworkMT.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

bool LoadMNIST(const char * path, vector<vector<float>> & data, vector<vector<float>> & targets, const unsigned int fCount, const char delim = ',')
{
	using namespace std;

	ifstream file(path);

	if (!file.is_open())
		return false;

	string line;
	string val;
	while (!file.eof())
	{
		getline(file, line);
		if (line.empty())
			return true;

		stringstream ss(line);

		
		getline(ss, val, delim);
		targets.push_back(vector<float>(10));
		targets.back()[stoi(val)] = 1;


		data.push_back(vector<float>(fCount));
		for (unsigned int i = 0; i < fCount; i++)
		{
			getline(ss, val, delim);
			data.back()[i] = stof(val);
		}
	}

	return true;
}

int main()
{
	vector<vector<float>> data;
	vector<vector<float>> targets;
	LoadMNIST("mnist_train.csv", data, targets, 28 * 28);

	vector<vector<float>> testdata;
	vector<vector<float>> testtargets;
	LoadMNIST("mnist_test.csv", testdata, testtargets, 28 * 28);

	
	NormalizeRowwise(data);
	NormalizeRowwise(testdata);

	NeuralNetworkMT nnmt({ 28 * 28 , 120 , 80 , 60 , 40 , 10 });

	nnmt.Master.Params.BatchSize = 32;
	nnmt.Master.Params.L1 = 0.00001f;
	nnmt.Master.Params.L2 = 0.00001f;
	nnmt.Master.Params.LearningRate = 0.0005f;
	nnmt.Master.Params.Momentum = 0.8f;

	while(true)
	{
		nnmt.Train(data, targets);

		std::cout << nnmt.Master.SquareError(data, targets)
			<< "\tIn:" << nnmt.Master.Accuracy(data, targets)
			<< "\tOut:" << nnmt.Master.Accuracy(testdata, testtargets)
			<< std::endl;
	}
}
