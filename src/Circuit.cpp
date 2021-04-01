#include "Circuit.hpp"
#include "Operators/operators.hpp"
#include "Variable.hpp"
#include "Univariate.hpp"


namespace qunn
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
	dim_ = rhs.dim_;
	states_from_left_ = rhs.states_from_left_;
	states_updated_to_ = rhs.states_updated_to_;
	for(const auto& op: rhs.ops_)
	{
		ops_.emplace_back(op->clone());
	}

	return *this;
}


Circuit operator|(Circuit a, const Circuit& b)
{
	a |= b;
	return a;
}

std::vector<Variable> Circuit::parameters() const
{
	std::vector<Variable> params;
	for(const auto& op: ops_)
	{
		if(auto diff_op = dynamic_cast<Univariate*>(op.get()))//may change?
		{
			params.emplace_back(diff_op->parameter());
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

			diff_op->parameter().add_grad(std::move(circuit_der));
		}
	}
}



} //namespace qunn
