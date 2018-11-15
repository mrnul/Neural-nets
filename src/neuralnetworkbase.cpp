#include <MyHeaders/neuralnetworkbase.h>

namespace neuralnetworkbase
{
	void NNBase::InitializeBase(vector<int> topology, const int threadCount)
	{
		const auto numOfLayers = topology.size();
		const auto lastIndex = numOfLayers - 1;

		#ifdef _OPENMP
		if (threadCount)
			omp_set_num_threads(threadCount);
		#endif

		D.resize(numOfLayers);
		Ex.resize(numOfLayers);
		O.resize(numOfLayers);
		Matrices.resize(numOfLayers);
		Grad.resize(numOfLayers);
		PrevGrad.resize(numOfLayers);

		for (int i = 1; i < numOfLayers; i++)
		{
			//these don't need a bias node
			D[i].setZero(1, topology[i]);
			Ex[i].setZero(1, topology[i]);

			//these need a bias node
			O[i - 1].setZero(1, topology[i - 1] + 1);
			O[i - 1](topology[i - 1]) = 1;

			//+1 for the bias of the prev layer
			//initialize Matrices with random numbers
			Matrices[i].setRandom(topology[i - 1] + 1, topology[i]);
			Matrices[i].topRows(Matrices[i].rows() - 1) *= sqrt(12.0f / (topology[i - 1] + 1.0f + topology[i]));
			//set biases to zero
			Matrices[i].row(Matrices[i].rows() - 1).setZero();

			//initialize Grad with value 0
			Grad[i].setZero(topology[i - 1] + 1, topology[i]);
			PrevGrad[i].setZero(topology[i - 1] + 1, topology[i]);
		}

		//last layer does not have a bias node
		O[lastIndex].setZero(1, topology[lastIndex]);
	}

	void NNBase::InitializeIndexVector(const int size)
	{
		Index.resize(size);
		for (int i = 0; i < size; i++)
			Index[i] = i;
	}

	void NNBase::ShuffleIndexVector()
	{
		std::random_shuffle(Index.begin(), Index.end());
	}

	void NNBase::SwapAndZeroGrad()
	{
		std::swap(Grad, PrevGrad);

		const auto GradSize = Grad.size();
		for (int l = 1; l < GradSize; l++)
			Grad[l].setZero();
	}

	bool NNBase::WriteWeightsToFile(const char * path) const
	{
		std::ofstream file(path, std::ios::out | std::ios::binary);
		if (!file.is_open())
			return false;

		const auto matricesCount = Matrices.size();
		for (int m = 1; m < matricesCount; m++)
		{
			const auto curSize = Matrices[m].size() * sizeof(float);
			file.write((const char*)Matrices[m].data(), curSize);
		}

		return true;
	}

	bool NNBase::LoadWeightsFromFile(const char * path)
	{
		std::ifstream file(path, std::ios::in | std::ios::binary);
		if (!file.is_open())
			return false;

		const auto matricesCount = Matrices.size();
		for (int m = 1; m < matricesCount; m++)
		{
			const auto curSize = Matrices[m].size() * sizeof(float);
			file.read((char*)Matrices[m].data(), curSize);
		}

		return true;
	}

	void NNBase::Dropout(MatrixXf & layer, const float p)
	{
		if (p == 0.f)
			return;

		//-1 to ignore last neuron (don't dropout the bias)
		const auto N = layer.size() - 1;
		for (int n = 0; n < N; n++)
		{
			const float rnd = (float)rand() / RAND_MAX;
			if (rnd <= p)
				layer(n) = 0.f;
		}
	}

	void NNBase::AddL1L2(const NNParams & params)
	{
		//Grad = Grad + l1term + l2term
		//don't regularize the bias
		//topRows(Grad[l].rows() - 1) skips the last row (the biases)

		const auto lCount = Grad.size();

		//both l1 and l2
		if (params.L1 != 0.f && params.L2 != 0.f)
		{
			for (int l = 1; l < lCount; l++)
				Grad[l].topRows(Grad[l].rows() - 1) +=
				params.L1 * Matrices[l].topRows(Grad[l].rows() - 1).unaryExpr(&Sign)
				+ params.L2 * Matrices[l].topRows(Grad[l].rows() - 1);
		}
		//only l1
		else if (params.L1 != 0.f)
		{
			for (int l = 1; l < lCount; l++)
				Grad[l].topRows(Grad[l].rows() - 1) += params.L1 * Matrices[l].topRows(Grad[l].rows() - 1).unaryExpr(&Sign);
		}
		//only l2
		else if (params.L2 != 0.f)
		{
			for (int l = 1; l < lCount; l++)
				Grad[l].topRows(Grad[l].rows() - 1) += params.L2 * Matrices[l].topRows(Grad[l].rows() - 1);
		}
	}

	void NNBase::AddMomentum(const NNParams & params)
	{
		if (params.Momentum == 0.f)
			return;

		const auto lCount = Grad.size();
		
		for (auto l = 0; l < lCount; l++)
			Grad[l] += PrevGrad[l] *params.Momentum;
	}

	void NNBase::UpdateWeights(const NNParams & params)
	{
		float norm = 1.f;
		//find the norm of the gradient
		if (params.NormalizeGradient)
		{
			float tmp = 0;
			const auto size = Grad.size();

			for (int i = 0; i < size; i++)
				tmp = tmp + Grad[i].squaredNorm();

			norm = sqrt(tmp);
		}

		const float coeff = params.LearningRate / norm;
		const auto MatricesSize = Matrices.size();

		//now each step has a length of 'LearningRate'
		for (int l = 1; l < MatricesSize; l++)
			Matrices[l] -= coeff * Grad[l];

		//apply max-norm regularization only if MaxNorm is specified
		if (params.MaxNorm == 0.f)
			return;

		//max-norm regularization
		for (int l = 1; l < MatricesSize; l++)
		{
			//column N has the weights of neuron N
			//find norms of all columns without taking into account their last element (the biase)
			//should we take into account the biases?
			const auto norms = Matrices[l].topRows(Matrices[l].rows() - 1).colwise().norm().eval();
			const auto count = norms.size();
			//iterate through each norm
			for (int n = 0; n < count; n++)
			{
				//and normalize if norm > MaxNorm
				if (norms(n) > params.MaxNorm)
					Matrices[l].topRows(Matrices[l].rows() - 1).col(n) *= params.MaxNorm / norms(n);
			}
		}
	}

	const MatrixXf & NNBase::Feedforward(const vector<float> & input, const NNParams & params)
	{
		std::copy(input.data(), input.data() + input.size(), O[0].data());

		const int lastDropoutIndex = (int)params.DropoutRates.size() - 1;
		if (lastDropoutIndex >= 0)
			Dropout(O[0], params.DropoutRates[0]);

		
		const int lastIndex = (int)Matrices.size() - 1;
		for (int m = 1; m < lastIndex; m++)
		{
			Ex[m].noalias() = O[m - 1] * Matrices[m];
			O[m].leftCols(O[m].size() - 1) = Ex[m].unaryExpr(&neuralnetworkbase::functions::ELU);

			if (m <= lastDropoutIndex)
				Dropout(O[m], params.DropoutRates[m]);
		}

		Ex[lastIndex].noalias() = O[lastIndex - 1] * Matrices[lastIndex];
		O[lastIndex] = Ex[lastIndex].unaryExpr([&](const float x) { return neuralnetworkbase::functions::Softmax(x, Ex[lastIndex]); });

		return O.back();
	}

	void NNBase::Backprop(const vector<float> & target)
	{
		const auto L = D.size() - 1;
		const auto outSize = O[L].size();

		//delta for output
		for (int j = 0; j < outSize; j++)
			D[L](j) = (O[L](j) - target[j]);

		//grad for output
		Grad[L].noalias() += O[L - 1].transpose() * D[L];

		//delta for hidden
		for (auto l = L; l > 1; l--)
		{
			//calc the sums
			//(Matrices * D.transpose()).transpose() is faster than D * Matrices.transpose()
			D[l - 1].noalias() = (Matrices[l].topRows(D[l - 1].size()) * D[l].transpose()).transpose();

			//multiply with derivatives
			D[l - 1] = D[l - 1].cwiseProduct(Ex[l - 1].unaryExpr(&neuralnetworkbase::functions::ELUDerivative));

			//grad for hidden
			Grad[l - 1].noalias() += O[l - 2].transpose() * D[l - 1];
		}
	}

	void NormalizeVector(vector<float> & vec, const float a, const float b)
	{
		float min = vec[0];
		float max = vec[0];
		const auto size = vec.size();
		for (int i = 1; i < size; i++)
		{
			if (min > vec[i])
				min = vec[i];
			else if (max < vec[i])
				max = vec[i];
		}

		const float denom = max - min;
		if (denom == 0.0f)
		{
			if (max > b)
			{
				for (int i = 0; i < size; i++)
					vec[i] = b;
			}
			else if (max < a)
			{
				for (int i = 0; i < size; i++)
					vec[i] = a;
			}

			return;
		}

		const float coeff = b - a;
		for (int i = 0; i < size; i++)
			vec[i] = coeff * (vec[i] - min) / denom + a;
	}

	void NormalizeColumnwise(vector<vector<float>> & data, const float a, const float b)
	{
		const auto DataCount = data.size();
		const auto FeatureCount = data[0].size();

		for (int f = 0; f < FeatureCount; f++)
		{
			float min = data[0][f];
			float max = data[0][f];
			for (int d = 1; d < DataCount; d++)
			{
				if (min > data[d][f])
					min = data[d][f];
				else if (max < data[d][f])
					max = data[d][f];
			}

			const float denom = max - min;
			if (denom == 0.0f)
			{
				if (max > b)
				{
					for (int d = 0; d < DataCount; d++)
						data[d][f] = b;
				}
				else if (max < a)
				{
					for (int d = 0; d < DataCount; d++)
						data[d][f] = a;
				}
				continue;
			}

			const float coeff = b - a;
			for (int d = 0; d < DataCount; d++)
				data[d][f] = coeff * (data[d][f] - min) / denom + a;
		}

	}

	void NormalizeRowwise(vector<vector<float>> & data, const float a, const float b)
	{
		const auto size = data.size();
		for (int i = 0; i < size; i++)
			NormalizeVector(data[i], a, b);
	}

	void StandarizeVector(vector<float> & vec)
	{
		float mean = 0;
		float sd = 0;
		const auto size = vec.size();

		for (int i = 0; i < size; i++)
			mean += vec[i];
		mean /= size;

		for (int i = 0; i < size; i++)
			sd += (vec[i] - mean) * (vec[i] - mean);
		sd /= size - 1;
		sd = sqrt(sd);

		for (int i = 0; i < size; i++)
			vec[i] = (vec[i] - mean) / sd;
	}

	void StandarizeVectors(vector<vector<float>> & data)
	{
		const auto size = data.size();
		for (int i = 0; i < size; i++)
			StandarizeVector(data[i]);
	}

	bool OneHotEncDec(const char * path, map<unsigned char, int> & enc, map<int, unsigned char> & dec, vector<__int16> & data)
	{
		ifstream file(path, std::ios::in | std::ios::binary);
		if (!file.is_open())
			return false;

		//the ID of the vector [0, 0, 0, 1, 0, 0] is 3
		//ID begins on 0 if maps are empty
		//else ID is the last + 1
		file.seekg(1077);
		int ID = dec.empty() ? 0 : (--dec.end())->first + 1;
		while (true)
		{
			unsigned char tmp;
			if (!file.read((char *)&tmp, 1))
				break;

			//if element does not exist in maps add it
			if (enc.find(tmp) == enc.end())
			{
				enc[tmp] = ID;
				dec[ID] = tmp;
				ID++;
			}

			//add byte in data vector
			data.push_back(tmp);
		}

		//256 means end of file
		data.push_back(256);
		return true;
	}

	int IndexOfMax(const MatrixXf & v)
	{
		float max = v(0);
		int iMax = 0;
		for (int i = 1; i < v.size(); i++)
		{
			if (max < v(i))
			{
				max = v(i);
				iMax = i;
			}
		}
		return iMax;
	}

	int IndexOfMax(const vector<float> & v)
	{
		float max = v[0];
		int iMax = 0;
		for (int i = 1; i < v.size(); i++)
		{
			if (max < v[i])
			{
				max = v[i];
				iMax = i;
			}
		}
		return iMax;
	}
}
