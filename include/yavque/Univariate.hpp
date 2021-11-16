#pragma once
#include "Operators/Operator.hpp"
#include "Variable.hpp"

namespace yavque
{
class Univariate
{
protected:
	Variable var_{0.0};


	Univariate(const Univariate& ) = default;
	Univariate(Univariate&& ) = default;

public:
	explicit Univariate() = default;

	explicit Univariate(Variable var)
		: var_{std::move(var)}
	{
	}

	Univariate& operator=(const Univariate& ) = delete;
	Univariate& operator=(Univariate&& ) = delete;
	virtual ~Univariate() = default;

	[[nodiscard]] Variable get_variable() const
	{
		return var_;
	}

	void change_variable(Variable var)
	{
		var_ = std::move(var);
	}

	[[nodiscard]] virtual std::unique_ptr<Operator> log_deriv() const = 0;
};
} //namespace yavque
