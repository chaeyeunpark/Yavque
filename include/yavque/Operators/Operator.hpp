#pragma once
#include <memory>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include "../Variable.hpp"
#include "../Counter.hpp"

namespace yavque
{
class Operator
{
public:
	virtual std::unique_ptr<Operator> clone() const = 0;

	uint32_t dim() const
	{
		return dim_;
	}

	virtual bool can_merge(const Operator& /* rhs */) const
	{
		return false;
	}
	/**
	 * return Op * st
	 */
	virtual Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const = 0;

	void dagger_in_place()
	{
		dagger_in_place_impl();
	}

	std::string name() const 
	{
		return name_;
	}

	virtual std::string desc() const
	{
		std::ostringstream ss;
		ss << "[" << name() << "]";
		return ss.str();
	}

protected:
	const uint32_t dim_;

private:
	std::string name_;

protected:

	explicit Operator(uint32_t dim, const std::string& name)
		: dim_{dim}, name_{name}
	{
		static Counter name_counter;
		if(name.empty())
		{
			//change to c++20 format
			std::stringstream ss;
			ss << "operator_";
			ss << name_counter.count();
			name_ = ss.str();
		}
	}

	void set_name(std::string new_name)
	{
		name_ = new_name;
	}

	virtual void dagger_in_place_impl() = 0;

public:
	virtual ~Operator() {}
};


inline Eigen::VectorXcd operator*(const Operator& op, 
		const Eigen::VectorXcd& st)
{
	return op.apply_right(st);
}


} // namespace yavque
