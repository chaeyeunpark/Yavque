#pragma once
#include <limits>
#include "Optimizer.hpp"

namespace yavque
{
class AdaMax
	: public Optimizer
{
public:
	static constexpr double DEFAULT_PARAMS[] = {0.002, 0.9, 0.999};

private:
	double alpha_;
	double beta1_;
	double beta2_;
	
	int t_;

	Eigen::VectorXd m_;
	Eigen::VectorXd u_;

public:

	AdaMax(double alpha = DEFAULT_PARAMS[0], double beta1 = DEFAULT_PARAMS[1],
			double beta2 = DEFAULT_PARAMS[2])
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), t_{0}
	{
	}

	AdaMax(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta1_(params.value("beta1", DEFAULT_PARAMS[1])),
			beta2_(params.value("beta2", DEFAULT_PARAMS[2])),
			t_{0}
	{
	}

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "AdaMax"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta1", DEFAULT_PARAMS[1]},
			{"beta2", DEFAULT_PARAMS[2]}
		};
	}

	nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "AdaMax"},
			{"alhpa", alpha_},
			{"beta1", beta1_},
			{"beta2", beta2_}
		};

	}

	Eigen::VectorXd getUpdate(const Eigen::VectorXd& grad) override
	{
		using std::pow;
		if(t_ == 0)
		{
			m_ = Eigen::VectorXd::Zero(grad.rows());
			u_ = Eigen::VectorXd::Zero(grad.rows());
		}
		++t_;
		m_ *= beta1_;
		m_ += (1.0-beta1_)*grad;

		u_ *= beta2_;
		u_ = u_.cwiseMax(grad.cwiseAbs());
		u_ = u_.cwiseMax(std::numeric_limits<double>::min());

		return -(alpha_/(1-pow(beta1_,t_)))*m_.cwiseQuotient(u_);
	}

};
}//namespace yavque
