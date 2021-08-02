#pragma once
#include <string>
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace yavque
{
class Circuit;
class VariableImpl
{
private:
	double value_;
	std::string name_;

	std::vector<std::shared_ptr<Circuit> > deriv_circuits_;
	mutable std::shared_ptr<Eigen::VectorXcd> grad_;
	mutable bool grad_updated_ = false;

public:
	VariableImpl(double value);
	VariableImpl(double value, std::string name);

	double value() const;
	void set_value(double new_value);

	std::string name() const;
	void set_name(std::string name);

	std::string desc() const;

	void add_grad(const Circuit& circ);
	void add_grad(Circuit&& circ);
	std::shared_ptr<const Eigen::VectorXcd> grad() const;
	void zero_grad();
};
} //namespace yavque
