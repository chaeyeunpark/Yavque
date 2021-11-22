#pragma once
#include <memory>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "../Counter.hpp"
#include "../Variable.hpp"

namespace yavque
{
class Operator
{
protected:
	uint32_t dim_;

private:
	std::string name_;

protected:
	explicit Operator(uint32_t dim, const std::string& name) : dim_{dim}, name_{name}
	{
		static Counter name_counter;
		if(name.empty())
		{
			// change to c++20 format
			std::stringstream ss;
			ss << "operator_";
			ss << name_counter.count();
			name_ = ss.str();
		}
	}

	void set_name(std::string new_name) { name_ = std::move(new_name); }

	virtual void dagger_in_place_impl() = 0;

public:
	Operator(const Operator&) = default;
	Operator(Operator&&) = default;

	Operator& operator=(const Operator&) = default;
	Operator& operator=(Operator&&) = default;

	[[nodiscard]] virtual std::unique_ptr<Operator> clone() const = 0;

	[[nodiscard]] uint32_t dim() const { return dim_; }

	[[nodiscard]] virtual bool can_merge(const Operator& /* rhs */) const
	{
		return false;
	}
	/**
	 * return Op * st
	 */
	[[nodiscard]] virtual Eigen::VectorXcd
	apply_right(const Eigen::VectorXcd& st) const = 0;

	void dagger_in_place() { dagger_in_place_impl(); }

	[[nodiscard]] std::string name() const { return name_; }

	[[nodiscard]] virtual std::string desc() const
	{
		std::ostringstream ss;
		ss << "[" << name() << "]";
		return ss.str();
	}

	virtual ~Operator() = default;
};

inline Eigen::VectorXcd operator*(const Operator& op, const Eigen::VectorXcd& st)
{
	return op.apply_right(st);
}

} // namespace yavque
