#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include "Variable.hpp"
#include "utilities.hpp"
#include "Univariate.hpp"

#include "Operators/Operator.hpp"
#include "Operators/SumLocalHam.hpp"

namespace qunn
{

class ProductHamEvol final
	: public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::SumLocalHamImpl> ham_;

	void dagger_in_place_impl() override
	{
		conjugate_ = !conjugate_;
	}

	ProductHamEvol(ProductHamEvol&& ) = default;
	ProductHamEvol(const ProductHamEvol& ) = default;

public:
	explicit ProductHamEvol(SumLocalHam ham)
		: Operator(ham.dim(), "ProductHamEvol of " + ham.name()), 
		ham_{ham.get_impl()}
	{
	}

	explicit ProductHamEvol(SumLocalHam ham, Variable var)
		: Operator(ham.dim(), "ProductHamEvol of " + ham.name()), 
		Univariate(std::move(var)), ham_{ham.get_impl()}
	{
	}

	ProductHamEvol& operator=(const ProductHamEvol& ) = delete;
	ProductHamEvol& operator=(ProductHamEvol&& ) = delete;

	std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<ProductHamEvol>{new ProductHamEvol(*this)};
		p->change_parameter(Variable{var_.value()});
		return p;
	}

	std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0.,1.0);
		std::string op_name = std::string("derivative of ") + name(); //change to fmt
		cx_double constant = conjugate_?I:-I;
		return std::make_unique<SumLocalHam>(ham_, op_name, constant);
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0.,1.0);
		Eigen::VectorXcd res = st;
		Eigen::MatrixXcd m = ham_->get_local_ham();

		double t = conjugate_?-var_.value():var_.value();
		Eigen::MatrixXcd expm = ham_->local_ham_exp(-I*t);

		for(uint32_t k = 0; k < ham_->num_qubits(); ++k)
		{
			res = apply_single_qubit(res, expm, k);
		}
		return res;
	}	

	bool can_merge(const Operator& rhs) const override
	{
		if(const ProductHamEvol* p = dynamic_cast<const ProductHamEvol*>(&rhs))
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
