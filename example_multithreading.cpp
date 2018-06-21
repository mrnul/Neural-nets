#include <MyHeaders\NeuralNetworkMT.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

bool LoadMNIST(const char * path, vector<vector<float>> & data, vector<vector<float>> & targets)
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

		getline(ss, val, ',');
		targets.push_back(vector<float>(10));
		targets.back()[stoi(val)] = 1;


		data.push_back(vector<float>(28 * 28));
		for (unsigned int i = 0; i < 28 * 28; i++)
		{
			getline(ss, val, ',');
			data.back()[i] = stof(val);
		}
	}

	return true;
}

int main()
{
	vector<vector<float>> data;
	vector<vector<float>> targets;
	LoadMNIST("mnist_train.csv", data, targets);

	vector<vector<float>> testdata;
	vector<vector<float>> testtargets;
	LoadMNIST("mnist_test.csv", testdata, testtargets);

	
	NormalizeRowwise(data);
	NormalizeRowwise(testdata);

	NeuralNetworkMT nnmt({ 28 * 28, 200, 150, 100, 80, 10 });

	nnmt.Master.Params.BatchSize = 30;
	nnmt.Master.Params.L1 = 0.000001f;
	nnmt.Master.Params.L2 = 0.0001f;
	nnmt.Master.Params.LearningRate = 0.1f;
	nnmt.Master.Params.Momentum = 0.8f;

	while(true)
	{
		//run 5 epochs
		for (int i = 0; i < 5; i++)
		{
			nnmt.Train(data, targets);
			nnmt.Master.Params.LearningRate *= 0.99f;
		}

		//show results
		std::cout << nnmt.Master.SquareError(data, targets)
			<< "\tIn:" << nnmt.Master.Accuracy(data, targets)
			<< "\tOut:" << nnmt.Master.Accuracy(testdata, testtargets)
			<< std::endl;
	}
}
