#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Optimizer.hpp"

namespace yavque
{
class SGDMomentum
	: public Optimizer
{
public:
	static constexpr double DEFAULT_PARAMS[] = {0.01, 0.0, 0.9};

private:
	double alpha_;
	double p_;
	double gamma_;

	Eigen::VectorXd m_;
	int t_;

public:
	SGDMomentum(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1],
			double gamma = DEFAULT_PARAMS[2])
		: alpha_{alpha}, p_{p}, gamma_{gamma}, t_{0}
	{
	}

	SGDMomentum(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])},
			gamma_{params.value("gamma", DEFAULT_PARAMS[2])},
			t_{0}
	{
	}

	nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"gamma", gamma_},
			{"p", p_}
		};
	}

	Eigen::VectorXd getUpdate(const Eigen::VectorXd& v) override
	{
		using std::pow;
		if(t_ == 0)
		{
			m_ = Eigen::VectorXd::Zero(v.size());
		}

		++t_;
		m_ *= gamma_;
		m_ += (1-gamma_)*v;
		double eta = std::max((alpha_/pow(t_, p_)), 1e-4)/(1.0 - pow(gamma_, t_));
		return -eta*m_;
	}
};
} //namespace yavque
