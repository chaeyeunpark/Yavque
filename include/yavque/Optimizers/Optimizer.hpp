#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace yavque
{
class Optimizer
{
public:
	Optimizer() = default;
	Optimizer(const Optimizer&) = default;
	Optimizer(Optimizer&&) = default;

	Optimizer& operator=(const Optimizer&) = default;
	Optimizer& operator=(Optimizer&&) = default;

	[[nodiscard]] virtual nlohmann::json desc() const = 0;

	virtual Eigen::VectorXd getUpdate(const Eigen::VectorXd& grad) = 0;

	virtual ~Optimizer() = default;
};
} // namespace yavque
