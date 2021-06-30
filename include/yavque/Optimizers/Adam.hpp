#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Optimizer.hpp"

namespace qunn
{
class Adam
	: public Optimizer
{
private:
	const double alpha_;
	const double beta1_;
	const double beta2_;
	const double eps_;

	int t_;
	Eigen::VectorXd m_;
	Eigen::VectorXd v_;

public:
	static constexpr double DEFAULT_PARAMS[] = {1e-3, 0.9, 0.999, 1e-8};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "Adam"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta1", DEFAULT_PARAMS[1]},
			{"beta2", DEFAULT_PARAMS[2]},
			{"eps", DEFAULT_PARAMS[3]},
		};
	}

	nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "Adam"},
			{"alhpa", alpha_},
			{"beta1", beta1_},
			{"beta2", beta2_},
			{"eps", eps_},
		};
	}

	Adam(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta1_(params.value("beta1", DEFAULT_PARAMS[1])),
			beta2_(params.value("beta2", DEFAULT_PARAMS[2])),
			eps_(params.value("eps", DEFAULT_PARAMS[3])),
			t_{0}
	{
	}

	Adam(double alpha = DEFAULT_PARAMS[0], double beta1 = DEFAULT_PARAMS[1],
			double beta2 = DEFAULT_PARAMS[2], double eps = DEFAULT_PARAMS[3])
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps), t_{0}
	{
	}

	Eigen::VectorXd getUpdate(const Eigen::VectorXd& grad) override
	{
		if(t_ ==0)
		{
			m_ = Eigen::VectorXd::Zero(grad.rows());
			v_ = Eigen::VectorXd::Zero(grad.rows());
		}
		++t_;

		m_ *= beta1_;
		m_ += (1-beta1_)*grad;

		Eigen::VectorXd g2 = grad.array().square();
		v_ *= beta2_;
		v_ += (1-beta2_)*g2;

		double epsnorm = eps_*sqrt(1.0-pow(beta2_,t_));
		Eigen::VectorXd denom = v_.unaryExpr([epsnorm](double x){
			return sqrt(x)+epsnorm; 
		});

		double alphat = alpha_*sqrt(1.0-pow(beta2_,t_))/(1.0-pow(beta1_,t_));

		return -alphat*m_.cwiseQuotient(denom);
	}
};
} //namespace qunn
