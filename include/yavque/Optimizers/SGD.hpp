#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Optimizer.hpp"

namespace yavque
{
class SGD
	: public Optimizer
{
public:
	static constexpr double DEFAULT_PARAMS[] = {0.01, 0.0};

private:
	double alpha_;
	double p_;
	int t_;

public:
	SGD(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1])
		: alpha_{alpha}, p_{p}, t_{0}
	{
	}

	SGD(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])},
			t_{0}
	{
	}

	nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"p", p_}
		};
	}

	Eigen::VectorXd getUpdate(const Eigen::VectorXd& v) override
	{
		using std::pow;
		++t_;
		double eta = std::max((alpha_/pow(t_, p_)), 1e-4);
		return -eta*v;
	}
};
}//namespace yavque
