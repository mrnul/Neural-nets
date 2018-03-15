#pragma once

#include <random>

using std::random_device;
using std::default_random_engine;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

class NormalRealRandom
{
	private:
		default_random_engine engine;
		normal_distribution<double> d;
	public:
		NormalRealRandom()
		{
			d.param(normal_distribution<double>::param_type());
			Seed();
		}

		NormalRealRandom(const double mean, const double sd)
		{
			d.param(normal_distribution<double>::param_type(mean, sd));
			Seed();
		}

		void SetParams(const double mean, const double sd)
		{
			d.param(normal_distribution<double>::param_type(mean, sd));
		}

		double Mean() const
		{
			return d.mean();
		}

		double SD() const
		{
			return d.stddev();
		}

		void Seed()
		{
			random_device rd;
			engine.seed(rd());
		}

		double operator()()
		{
			return d(engine);
		}
};

class UniformIntRandom
{
	private:
		default_random_engine engine;
		uniform_int_distribution<int> d;
	public:
		UniformIntRandom()
		{
			d.param(uniform_int_distribution<int>::param_type(0, 1));
			Seed();
		}

		UniformIntRandom(const int a, const int b)
		{
			d.param(uniform_int_distribution<int>::param_type(a, b));
			Seed();
		}

		void SetParams(const int a, const int b)
		{
			d.param(uniform_int_distribution<int>::param_type(a, b));
		}

		int a() const
		{
			return d.a();
		}

		int b() const
		{
			return d.b();
		}

		void Seed()
		{
			random_device rd;
			engine.seed(rd());
		}

		int operator()()
		{
			return d(engine);
		}
};

class UniformRealRandom
{
	private:
		default_random_engine engine;
		uniform_real_distribution<double> d;
	public:
		UniformRealRandom()
		{
			d.param(uniform_real_distribution<double>::param_type(0, 1));
			Seed();
		}

		UniformRealRandom(const double a, const double b)
		{
			d.param(uniform_real_distribution<double>::param_type(a, b));
			Seed();
		}

		void SetParams(const double a, const double b)
		{
			d.param(uniform_real_distribution<double>::param_type(a, b));
		}

		double a() const
		{
			return d.a();
		}

		double b() const
		{
			return d.b();
		}

		void Seed()
		{
			random_device rd;
			engine.seed(rd());
		}

		double operator()()
		{
			return d(engine);
		}
};
