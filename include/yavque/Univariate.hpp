#pragma once
#include "Variable.hpp"
#include "Operators/Operator.hpp"

namespace yavque
{
class Univariate
{
protected:
	Variable var_ = 0.0;

public:

	explicit Univariate() = default;
	explicit Univariate(Variable var)
		: var_{std::move(var)}
	{
	}

	Variable parameter()
	{
		return var_;
	}

	const Variable parameter() const
	{
		return var_;
	}

	void change_parameter(Variable var)
	{
		var_ = std::move(var);
	}

	virtual std::unique_ptr<Operator> log_deriv() const = 0;
};
} //namespace yavque
