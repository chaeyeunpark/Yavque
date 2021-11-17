#pragma once
#include "Optimizer.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <nlohmann/json.hpp>

namespace yavque
{
class SGD
	: public Optimizer
{
public:
	static constexpr std::array<double, 3> DEFAULT_PARAMS = {0.01, 0.0, 1e-4};

private:
	const double alpha_;
	const double p_;
	const double min_alpha_;
	int t_ = 0;

public:
	explicit SGD(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1],
			double min_alpha = DEFAULT_PARAMS[2])
		: alpha_{alpha}, p_{p}, min_alpha_{min_alpha}
	{
	}

	explicit SGD(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])}, 
			min_alpha_{params.value("min_alpha", DEFAULT_PARAMS[2])}
	{
	}

	[[nodiscard]] nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"p", p_},
			{"min_alpha", min_alpha_}
		};
	}

	Eigen::VectorXd getUpdate(const Eigen::VectorXd& v) override
	{
		using std::pow;
		++t_;
		double eta = std::max((alpha_/pow(t_, p_)), min_alpha_);
		return -eta*v;
	}
};
}//namespace yavque
