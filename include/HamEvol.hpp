#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include "Operator.hpp"
#include "Variable.hpp"
#include "utility.hpp"
#include "Univerate.hpp"
#include "Hamiltonian.hpp"
#include "NonHermitian.hpp"

namespace qunn
{

class HamEvol final
	: public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<HamiltonianImpl> ham_;

	void dagger_in_place_impl() override
	{
		conjugate_ = true;
	}

	explicit HamEvol(std::shared_ptr<HamiltonianImpl> ham, std::string name, Variable var)
		: Operator(ham->get_ham().rows(), std::move(name)), 
		Univariate(std::move(var)), ham_{ham}
	{
	}


public:
	explicit HamEvol(Hamiltonian ham)
		: Operator(ham.dim(), "HamEvol of " + ham.name()), ham_{ham.get_impl()}
	{
	}

	explicit HamEvol(Hamiltonian ham, Variable var)
		: Operator(ham.dim(), "HamEvol of " + ham.name()), 
		Univariate(std::move(var)), ham_{ham.get_impl()}
	{
	}


	HamEvol(HamEvol&& ) = default;
	HamEvol(const HamEvol& ) = default;

	HamEvol& operator=(const HamEvol& ) = delete;
	HamEvol& operator=(HamEvol&& ) = delete;

	Hamiltonian hamiltonian() const
	{
		return Hamiltonian(ham_);
	}

	std::unique_ptr<Operator> clone() const override
	{
		return std::unique_ptr<HamEvol>{new HamEvol(ham_, name() + " copied", Variable{var_.value()})};
		//return std::make_unique<HamEvol>(ham_, name() + " copied", Variable{var_.value()});
	}

	std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0.,1.0);
		std::string op_name = std::string("derivative of ") + name(); //change to fmt
		return std::make_unique<NonHermitian>(op_name, -I*ham_->get_ham());
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0.,1.0);
		Eigen::VectorXcd res = ham_->evecs().adjoint()*st;
		double t = conjugate_?-var_.value():var_.value();
		res.array() *= exp(-I*ham_->evals().array()*t);
		return ham_->evecs()*res;
	}	

	bool can_merge(const Operator& rhs) const override
	{
		if(const HamEvol* p = dynamic_cast<const HamEvol*>(&rhs))
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
