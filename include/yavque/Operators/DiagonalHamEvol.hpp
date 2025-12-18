#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "../Univariate.hpp"
#include "../Variable.hpp"
#include "../utils.hpp"

#include "DiagonalOperator.hpp"
#include "Operator.hpp"

#include <memory>

namespace yavque
{

class DiagonalHamEvol final : public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::DiagonalOperatorImpl> ham_;

	void dagger_in_place_impl() override { conjugate_ = !conjugate_; }

	DiagonalHamEvol(const DiagonalHamEvol&) = default;
	DiagonalHamEvol(DiagonalHamEvol&&) = default;

public:
	explicit DiagonalHamEvol(const DiagonalOperator& ham)
		: Operator(ham.dim(), "DiagonalHamEvol of " + ham.name()), ham_{ham.get_impl()}
	{
	}

	explicit DiagonalHamEvol(const DiagonalOperator& ham, Variable var)
		: Operator(ham.dim(), "DiagonalHamEvol of " + ham.name()),
		  Univariate(std::move(var)), ham_{ham.get_impl()}
	{
	}

	DiagonalHamEvol& operator=(const DiagonalHamEvol&) = delete;
	DiagonalHamEvol& operator=(DiagonalHamEvol&&) = delete;

	~DiagonalHamEvol() override = default;

	[[nodiscard]] DiagonalOperator hamiltonian() const { return DiagonalOperator(ham_); }

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<DiagonalHamEvol>(new DiagonalHamEvol(*this));
		p->change_variable(Variable{var_.value()});
		return p;
	}

	[[nodiscard]] std::unique_ptr<Operator> log_deriv() const override
	{
		const std::string op_name
			= std::string("derivative of ") + name(); // change to fmt
		const cx_double constant = conjugate_ ? std::complex<double>(0.0, 1.0) : std::complex<double>(0.0, -1.0);
		return std::make_unique<DiagonalOperator>(ham_, op_name, constant);
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		const double t = conjugate_ ? -var_.value() : var_.value();
		return exp(ham_->get_diag_op().array() * std::complex<double>{0.0, -t}) * st.array();
	}

	[[nodiscard]] bool can_merge(const Operator& rhs) const override
	{
		if(const auto* p = dynamic_cast<const DiagonalHamEvol*>(&rhs))
		{
			if(ham_ == p->ham_)
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
} // namespace yavque
