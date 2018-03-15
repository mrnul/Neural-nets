#include <MyHeaders\Matrix.h>

Matrix::Matrix()
{
	InCount = OutCount = 0;
}

Matrix::Matrix(const unsigned int inputCount, const unsigned int ouputCount, const double min, const double max)
{
	Initialize(inputCount, ouputCount, min, max);
}

void Matrix::Initialize(const unsigned int inputCount, const unsigned int outputCount, const double min, const double max)
{
	if (inputCount == 0 || outputCount == 0)
		return;

	InCount = inputCount;
	OutCount = outputCount;

	const unsigned int total = inputCount * outputCount;
	M.resize(total);

	if (min == max)
		return;
	
	UniformRealRandom Random(min, max);
	for (unsigned int i = 0; i < total; i++)
			M[i] = Random();
}

unsigned int Matrix::InputCount() const
{
	return InCount;
}

unsigned int Matrix::OutputCount() const
{
	return OutCount;
}

double Matrix::Element(const unsigned int i, const unsigned int j) const
{
	return M[j * InCount + i];
}

double Matrix::Element(const unsigned int index) const
{
	return M[index];
}

void Matrix::Set(const unsigned int i, const unsigned int j, const double value)
{
	M[j * InCount + i] = value;
}

void Matrix::Set(const unsigned int index, const double value)
{
	M[index] = value;
}

void Matrix::Fill(const unsigned int value)
{
	std::fill(M.begin(), M.end(), 0);
}

const double * Matrix::Data() const
{
	return M.data();
}

unsigned int Matrix::WeightsCount() const
{
	return InCount * OutCount;
}

void VectorMatrixProduct(const vector<double> &vec, const Matrix &mat, vector<double> &product)
{
	for (unsigned int j = 0; j < mat.OutputCount(); j++)
	{
		double sum = 0;
		for (unsigned int i = 0; i < mat.InputCount(); i++)
			sum += vec[i] * mat.Element(i, j);

		product[j] = sum;
	}
}
