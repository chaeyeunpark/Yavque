#pragma once
#include <Eigen/Dense>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace yavque
{
class Circuit;
class VariableImpl
{
private:
	double value_;
	std::string name_;

	std::vector<std::shared_ptr<Circuit>> deriv_circuits_ = {};
	mutable std::shared_ptr<Eigen::VectorXcd> grad_ = {};
	mutable bool grad_updated_ = false;

public:
	explicit VariableImpl(double value) : value_{value}
	{
		static Counter counter;
		std::ostringstream ss; // change to format C++20
		ss << "variable " << counter.count();
		name_ = ss.str();
	}

	VariableImpl(double value, std::string name) : value_{value}, name_{std::move(name)}
	{
	}

	double value() const { return value_; }

	void set_value(double new_value) { value_ = new_value; }

	[[nodiscard]] std::string name() const { return name_; }

	void set_name(const std::string& name) { name_ = name; }

	[[nodiscard]] std::string desc() const;

	void add_grad(const Circuit& circ);
	void add_grad(Circuit&& circ);

	std::shared_ptr<const Eigen::VectorXcd> grad() const;
	void zero_grad();
};
} // namespace yavque
