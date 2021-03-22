#pragma once
#include <cstdint>
#include <memory>
#include <Eigen/Dense>

namespace qunn
{
class QuantumState
{
private:
	std::shared_ptr<Eigen::VectorXd> state_;

public:
	QuantumState(const Eigen::VectorXd& state)
		: state_{std::make_shared<Eigen::VectorXd>(state)}
	{
	}
};
}
