#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include "../Variable.hpp"
#include "../utils.hpp"
#include "../Univariate.hpp"

#include "Operator.hpp"
#include "DiagonalOperator.hpp"

namespace qunn
{

class DiagonalHamEvol final
	: public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::DiagonalOperatorImpl> ham_;

	void dagger_in_place_impl() override
	{
		conjugate_ = !conjugate_;
	}

	DiagonalHamEvol(const DiagonalHamEvol& ) = default;
	DiagonalHamEvol(DiagonalHamEvol&& ) = default;

public:
	explicit DiagonalHamEvol(DiagonalOperator ham)
		: Operator(ham.dim(), "DiagonalHamEvol of " + ham.name()), 
		ham_{ham.get_impl()}
	{
	}

	explicit DiagonalHamEvol(DiagonalOperator ham, Variable var)
		: Operator(ham.dim(), "DiagonalHamEvol of " + ham.name()), 
		Univariate(std::move(var)), ham_{ham.get_impl()}
	{
	}

	DiagonalHamEvol& operator=(const DiagonalHamEvol& ) = delete;
	DiagonalHamEvol& operator=(DiagonalHamEvol&& ) = delete;


	DiagonalOperator hamiltonian() const
	{
		return DiagonalOperator(ham_);
	}

	std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<DiagonalHamEvol>(new DiagonalHamEvol(*this));
		p->change_parameter(Variable{var_.value()});
		return p;
	}

	std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0.,1.0);
		std::string op_name = std::string("derivative of ") + name(); //change to fmt
		cx_double constant = conjugate_?I:-I;
		return std::make_unique<DiagonalOperator>(ham_, op_name, constant);
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0.,1.0);
		double t = conjugate_?-var_.value():var_.value();
		return exp(-I*ham_->get_diag_op().array()*t)*st.array();
	}	

	bool can_merge(const Operator& rhs) const override
	{
		if(const DiagonalHamEvol* p = dynamic_cast<const DiagonalHamEvol*>(&rhs))
		{
			if (ham_ == p->ham_)
				return true;
		}
		return false;
	}

	std::string desc() const override
	{
		std::ostringstream ss;
		ss << "[" << name() << ", variable name: " << var_.name() << ", " 
			<< "variable value: " << var_.value() << "]";
		return ss.str();
	}
};
}
