#include "yavque/Circuit.hpp"
#include "yavque/Univariate.hpp"
#include "yavque/Variable.hpp"

#include "yavque/operators.hpp"

namespace yavque
{
Circuit::Circuit(const Circuit& rhs)
	: dim_{rhs.dim_}, states_from_left_{rhs.states_from_left_},
	states_updated_to_{rhs.states_updated_to_}
{
	for(const auto& op: rhs.ops_)
	{
		ops_.emplace_back(op->clone());
	}
}

Circuit& Circuit::operator=(const Circuit& rhs)
{
	if(this == &rhs)
	{
		return *this;
	}
	dim_ = rhs.dim_;
	states_from_left_ = rhs.states_from_left_;
	states_updated_to_ = rhs.states_updated_to_;

	ops_.clear();
	for(const auto& op: rhs.ops_)
	{
		ops_.emplace_back(op->clone());
	}

	return *this;
}

std::vector<Variable> Circuit::variables() const
{
	std::vector<Variable> params;
	for(const auto& op: ops_)
	{
		if(auto* diff_op = dynamic_cast<Univariate*>(op.get()))//may change?
		{
			params.emplace_back(diff_op->get_variable());
		}
	}
	return params;
}

void Circuit::derivs() const
{
	for(uint32_t idx = 0; idx < ops_.size(); ++idx)
	{
		if(auto diff_op = dynamic_cast<Univariate*>(ops_[idx].get()))//may change?
		{
			Circuit circuit_der = this->to(idx+1);
			circuit_der.add_op_right(diff_op->log_deriv());
			circuit_der |= this->from(idx+1);

			diff_op->get_variable().add_grad(std::move(circuit_der));
		}
	}
}

Circuit operator|(Circuit a, const Circuit& b)
{
	a |= b;
	return a;
}
Circuit operator|(Circuit a, Circuit&& b)
{
	a |= std::move(b);
	return a;
}

} //namespace yavque
