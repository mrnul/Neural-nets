#pragma once

#include <MyHeaders\XRandom.h>
#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;

class Matrix
{
	private:
		unsigned int InCount;
		unsigned int OutCount;
		vector<double> M;
	public:

		Matrix();
		Matrix(const unsigned int inputCount, const unsigned int ouputCount, const double min = 0, const double max = 0);
		void Initialize(const unsigned int inputCount, const unsigned int ouputCount, const double min = 0, const double max = 0);
		unsigned int InputCount() const;
		unsigned int OutputCount() const;
		double Element(const unsigned int i, const unsigned int j) const;
		double Element(const unsigned int index) const;
		void Set(const unsigned int i, const unsigned int j, const double value);
		void Set(const unsigned int index, const double value);
		void Fill(const unsigned int value);
		const double * Data() const;
		unsigned int WeightsCount() const;
};

void VectorMatrixProduct(const vector<double> &vec, const Matrix &mat, vector<double> &product);
