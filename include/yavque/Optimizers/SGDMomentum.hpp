#pragma once
#include "Optimizer.hpp"

#include <Eigen/Dense>
#include <array>
#include <nlohmann/json.hpp>

namespace yavque
{
class SGDMomentum
	: public Optimizer
{
public:
	static constexpr std::array<double, 4> DEFAULT_PARAMS = {0.01, 0.0, 0.9, 1e-4};

private:
	double alpha_;
	double p_;
	double gamma_;
	double min_alpha_;

	Eigen::VectorXd m_;
	int t_ = 0;

public:
	explicit SGDMomentum(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1],
			double gamma = DEFAULT_PARAMS[2], double min_alpha = DEFAULT_PARAMS[3])
		: alpha_{alpha}, p_{p}, gamma_{gamma}, min_alpha_{min_alpha}
	{
	}

	explicit SGDMomentum(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])},
			gamma_{params.value("gamma", DEFAULT_PARAMS[2])},
			min_alpha_{params.value("min_alpha", DEFAULT_PARAMS[3])}
	{
	}

	[[nodiscard]] nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"gamma", gamma_},
			{"p", p_},
			{"min_alpha", min_alpha_}
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
		double eta = std::max((alpha_/pow(t_, p_)), min_alpha_)/(1.0 - pow(gamma_, t_));
		return -eta*m_;
	}
};
} //namespace yavque
