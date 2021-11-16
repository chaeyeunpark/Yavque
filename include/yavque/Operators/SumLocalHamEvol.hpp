#pragma once
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <Eigen/Sparse>

#include "../Univariate.hpp"
#include "../Variable.hpp"
#include "../utils.hpp"

#include "Operator.hpp"
#include "SumLocalHam.hpp"

namespace yavque
{

class SumLocalHamEvol final
	: public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::SumLocalHamImpl> ham_;

	void dagger_in_place_impl() override
	{
		conjugate_ = !conjugate_;
	}

	SumLocalHamEvol(SumLocalHamEvol&& ) = default;
	SumLocalHamEvol(const SumLocalHamEvol& ) = default;

public:
	explicit SumLocalHamEvol(const SumLocalHam& ham)
		: Operator(ham.dim(), "SumLocalHamEvol of " + ham.name()), 
		ham_{ham.get_impl()}
	{
	}

	explicit SumLocalHamEvol(const SumLocalHam& ham, Variable var)
		: Operator(ham.dim(), "SumLocalHamEvol of " + ham.name()), 
		Univariate(std::move(var)), ham_{ham.get_impl()}
	{
	}

	SumLocalHamEvol& operator=(const SumLocalHamEvol& ) = delete;
	SumLocalHamEvol& operator=(SumLocalHamEvol&& ) = delete;

	~SumLocalHamEvol() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<SumLocalHamEvol>{new SumLocalHamEvol(*this)};
		p->change_variable(Variable{var_.value()});
		return p;
	}

	[[nodiscard]] std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0.,1.0);
		std::string op_name = std::string("derivative of ") + name(); //change to fmt
		cx_double constant = conjugate_?I:-I;
		return std::make_unique<SumLocalHam>(ham_, op_name, constant);
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
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

	[[nodiscard]] bool can_merge(const Operator& rhs) const override
	{
		if(const auto* p = dynamic_cast<const SumLocalHamEvol*>(&rhs))
		{
			if (ham_ == p->ham_)
			{
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] std::string desc() const override
	{
		std::ostringstream ss;
		ss << "[" << name() << ", variable name: " << var_.name() << ", " 
			<< "variable value: " << var_.value() << "]";
		return ss.str();
	}
};
}// namespace yavque
