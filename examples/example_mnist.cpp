#include <MyHeaders/NeuralNetwork.h>
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

	neuralnetworkbase::StandarizeVectors(data);
	neuralnetworkbase::StandarizeVectors(testdata);

	NeuralNetwork nn({ 784 , 300, 300, 10 }, 2);

	nn.Params.BatchSize = 500;
	nn.Params.LearningRate = 0.2f;
	nn.Params.Momentum = 0.95f;
	//nn.Params.L1 = 0.01f;
	//nn.Params.L2 = 0.5f;
	nn.Params.DropoutRates = { 0.2f, 0.2f, 0.2f }; // dropout rate for each layer (don't apply drop out to the last layer)
	//nn.Params.MaxNorm = 3.f;
	nn.Params.NormalizeGradient = true;

	int epoch = 0;
	while (true)
	{
		nn.Train(data, targets);
		epoch++;

		std::cout
			<< nn.SquareError(data, targets)
			<< "\tTrain:" << nn.Accuracy(data, targets)
			<< "\tTest:" << nn.Accuracy(testdata, testtargets)
			<< "\tEpoch:" << epoch
			<< "\tLR:" << nn.Params.LearningRate
			<< std::endl;

		nn.Params.LearningRate *= 0.95f;
	}
}
