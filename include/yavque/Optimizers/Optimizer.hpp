#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace yavque
{
class Optimizer
{
public:
	virtual nlohmann::json desc() const = 0; 

	virtual Eigen::VectorXd getUpdate(const Eigen::VectorXd& grad) = 0;

	virtual ~Optimizer() { }
};
} //namespace yavque