#include <MyHeaders/neuralnetworkbase.h>

namespace neuralnetworkbase
{
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

	int OneHotEncDec(const char * path, map<char, vector<float>>& enc, map<vector<float>, char>& dec, vector<vector<float>>& data)
	{
		ifstream file(path);
		if (!file.is_open())
			return 0;

		while (!file.eof())
		{
			const int tmp = file.get();
			//if (!isprint(tmp))
				//continue;

			enc[tmp] = vector<float>();
		}

		int i = 0;
		for (auto & x : enc)
		{
			x.second = vector<float>(enc.size());
			x.second[i] = 1.f;
			dec[x.second] = x.first;
			i++;
		}

		file.clear();
		file.seekg(0);
		while (!file.eof())
		{
			const int tmp = file.get();
			//if (!isprint(tmp))
				//continue;
			data.push_back(enc[tmp]);
		}

		return (int)enc.size();
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

	void InitializeIndexVector(NNBase & base, const int size)
	{
		base.Index.resize(size);
		for (int i = 0; i < size; i++)
			base.Index[i] = i;
	}

	void ShuffleIndexVector(NNBase & base)
	{
		std::random_shuffle(base.Index.begin(), base.Index.end());
	}

	void DropOut(MatrixXf & O, const float DropOutRate)
	{
		if (DropOutRate == 0.f)
			return;

		//-1 to ignore last neuron (don't dropout the bias)
		const auto N = O.size() - 1;
		for (int n = 0; n < N; n++)
		{
			const float rnd = (float)rand() / RAND_MAX;
			if (rnd <= DropOutRate)
				O(n) = 0.f;
		}
	}

	void AddL1L2(NNBase & base, const float l1, const float l2)
	{
		//Grad = Grad + l1term + l2term
		//don't regularize the bias
		//topRows(Grad[l].rows() - 1) skips the last row (the biases)

		const auto lCount = base.Grad.size();

		//both l1 and l2
		if (l1 != 0.f && l2 != 0.f)
		{
			#pragma omp parallel for
			for (int l = 1; l < lCount; l++)
				base.Grad[l].topRows(base.Grad[l].rows() - 1) +=
				l1 * base.Matrices[l].topRows(base.Grad[l].rows() - 1).unaryExpr(&Sign)
				+ l2 * base.Matrices[l].topRows(base.Grad[l].rows() - 1);
		}
		//only l1
		else if (l1 != 0.f)
		{
			#pragma omp parallel for
			for (int l = 1; l < lCount; l++)
				base.Grad[l].topRows(base.Grad[l].rows() - 1) += l1 * base.Matrices[l].topRows(base.Grad[l].rows() - 1).unaryExpr(&Sign);
		}
		//only l2
		else if (l2 != 0.f)
		{
			#pragma omp parallel for
			for (int l = 1; l < lCount; l++)
				base.Grad[l].topRows(base.Grad[l].rows() - 1) += l2 * base.Matrices[l].topRows(base.Grad[l].rows() - 1);
		}
	}

	void AddMomentum(NNBase & base, const float momentum)
	{
		if (momentum == 0.f)
			return;

		const auto lCount = base.Grad.size();
		
		#pragma omp parallel for
		for (auto l = 0; l < lCount; l++)
			base.Grad[l] += base.PrevGrad[l] * momentum;
	}

	void UpdateWeights(NNBase & base, const float rate, const bool NormalizeGrad)
	{
		float norm = 1.f;
		if (NormalizeGrad)
		{
			float tmp = 0;
			const auto size = base.Grad.size();

			#pragma omp parallel for reduction (+:tmp)
			for (int i = 0; i < size; i++)
				tmp = tmp + base.Grad[i].squaredNorm();

			norm = sqrt(tmp);
		}

		const float coeff = rate / norm;
		const auto MatricesSize = base.Matrices.size();

		#pragma omp parallel for
		for (int l = 1; l < MatricesSize; l++)
			base.Matrices[l] -= coeff * base.Grad[l];
	}

	const MatrixXf & FeedForward(NNBase & base, const vector<float> & input, const float DropOutRate)
	{
		std::copy(input.data(), input.data() + input.size(), base.O[0].data());

		DropOut(base.O[0], DropOutRate);

		const auto lastIndex = base.Matrices.size() - 1;
		for (int m = 1; m < lastIndex; m++)
		{
			base.Ex[m].noalias() = base.O[m - 1] * base.Matrices[m];
			base.O[m].leftCols(base.O[m].size() - 1) = base.Ex[m].unaryExpr(&neuralnetworkbase::functions::ELU);

			DropOut(base.O[m], DropOutRate);
		}

		base.Ex[lastIndex].noalias() = base.O[lastIndex - 1] * base.Matrices[lastIndex];
		base.O[lastIndex] = base.Ex[lastIndex].unaryExpr([&](const float x) { return neuralnetworkbase::functions::Softmax(x, base.Ex[lastIndex]); });

		return base.O.back();
	}

	void BackProp(NNBase & base, const vector<float> & target)
	{
		const auto L = base.D.size() - 1;
		const auto outSize = base.O[L].size();

		//delta for output
		for (int j = 0; j < outSize; j++)
			base.D[L](j) = (base.O[L](j) - target[j]);

		//grad for output
		base.Grad[L].noalias() += base.O[L - 1].transpose() * base.D[L];

		//delta for hidden
		for (auto l = L; l > 1; l--)
		{
			//calc the sums
			//(Matrices * D.transpose()).transpose() is faster than D * Matrices.transpose()
			base.D[l - 1].noalias() = (base.Matrices[l].topRows(base.D[l - 1].size()) * base.D[l].transpose()).transpose();

			//multiply with derivatives
			base.D[l - 1] = base.D[l - 1].cwiseProduct(base.Ex[l - 1].unaryExpr(&neuralnetworkbase::functions::ELUDerivative));

			//grad for hidden
			base.Grad[l - 1].noalias() += base.O[l - 2].transpose() * base.D[l - 1];
		}
	}

	void FeedAndBackProp(NNBase & base, const vector<vector<float>> & inputs, const vector<vector<float>> & targets,
		const int start, const int end, const float DropOutRate)
	{
		for (int i = start; i < end; i++)
		{
			FeedForward(base, inputs[base.Index[i]], DropOutRate);
			BackProp(base, targets[base.Index[i]]);
		}
	}
}