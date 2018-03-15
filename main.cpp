#include <MyHeaders\NeuralNetwork.h>

int main()
{
	//XOR data
	vector<vector<double>> data = { { 1,1 },{ 1,0 },{ 0,1 },{ 0,0 } };
	vector<vector<double>> targets = { { 0 },{ 1 },{ 1 },{ 0 } };

	//1 input layer with 2 nodes
	//1 hidden layer with 2 nodes
	//1 output layer with 1 node
	vector<unsigned int> topology = { 2 ,2 ,1 };

	//create network with random weights in range [-0.5, 0.5]
	NeuralNetwork nn(topology, -0.5, 0.5);
	
	double error = 0;
	do{
		nn.Train(data, targets, 1);
		error = nn.Error(data, targets);
		cout << error << endl;
	} while (error > 0.0001);

	cout << "End" << endl;
	std::cin.get();
}
