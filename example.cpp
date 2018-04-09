#include <MyHeaders\NeuralNetwork.h>

int main()
{
	//XOR data
	vector<vector<float>> data = { { 0,0 },{ 1,0 },{ 0,1 },{ 1,1 } };
	vector<vector<float>> targets = { { 0 },{ 1 },{ 1 },{ 0 } };

	//create a neural network with 3 layers
	//2 input nodes, 2 hidden nodes, 1 output node
	NeuralNetwork nn({ 2 , 2 , 1 });

	while (true)
	{
		nn.Train(data, targets, 0.75f);
		std::cout << nn.SquareError(data, targets) << std::endl;
	}

}
