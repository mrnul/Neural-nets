#include <MyHeaders\NeuralNetworkMT.h>


int main()
{
	//XOR data
	vector<vector<float>> data = { { 0,0 },{ 1,0 },{ 0,1 },{ 1,1 } };
	vector<vector<float>> targets = { { 0 },{ 1 },{ 1 },{ 0 } };

	//create a neural network with 3 layers
	//2 input nodes, 2 hidden nodes, 1 output node
	//and use 2 threads
	NeuralNetworkMT nnmt({ 2 , 2 , 1 }, 2);

	while (true)
	{
		nnmt.Train(data, targets, 0.75f);
		std::cout << nnmt.SquareError(data, targets) << std::endl;
	}

}
