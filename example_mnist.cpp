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

	NeuralNetwork nn({ 784 , 250, 250, 10 }, 2);

	nn.Params.BatchSize = 100;
	nn.Params.LearningRate = 0.1f;
	nn.Params.Momentum = 0.8f;
	nn.Params.DropOutRate = 0.2f;
	nn.Params.NormalizeGradient = true;

	int epoch = 0;
	while (true)
	{
		nn.Train(data, targets);
		nn.Params.LearningRate *= 0.95f;
		epoch++;

		std::cout << nn.SquareError(data, targets)
			<< "\tIn:" << nn.Accuracy(data, targets)
			<< "\tOut:" << nn.Accuracy(testdata, testtargets)
			<< "\tEpoch:" << epoch
			<< std::endl;
	}
}