#pragma once

#include "../Univariate.hpp"
#include "../Variable.hpp"
#include "../utils.hpp"

#include "Hamiltonian.hpp"
#include "Operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <memory>

namespace yavque
{

class HamEvol final : public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::HamiltonianImpl> ham_;

	void dagger_in_place_impl() override { conjugate_ = !conjugate_; }

	HamEvol(HamEvol&&) = default;
	HamEvol(const HamEvol&) = default;

public:
	explicit HamEvol(const Hamiltonian& ham)
		: Operator(ham.dim(), "HamEvol of " + ham.name()), ham_{ham.get_impl()}
	{
	}

	explicit HamEvol(const Hamiltonian& ham, Variable var)
		: Operator(ham.dim(), "HamEvol of " + ham.name()), Univariate(std::move(var)),
		  ham_{ham.get_impl()}
	{
	}

	HamEvol& operator=(const HamEvol&) = delete;
	HamEvol& operator=(HamEvol&&) = delete;
	~HamEvol() override = default;

	[[nodiscard]] Hamiltonian hamiltonian() const { return Hamiltonian(ham_); }

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<HamEvol>{new HamEvol(*this)};
		p->change_variable(Variable{var_.value()});
		return p;
	}

	[[nodiscard]] std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0., 1.0);
		const std::string op_name
			= std::string("derivative of ") + name(); // change to fmt
		const cx_double constant = conjugate_ ? I : -I;
		return std::make_unique<Hamiltonian>(ham_, op_name, constant);
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0., 1.0);
		Eigen::VectorXcd res = ham_->evecs().adjoint() * st;
		const double t = [&]()
		{
			if(conjugate_)
			{
				return -var_.value();
			}
			return var_.value();
		}();
		res.array() *= exp(-I * ham_->evals().array() * t);
		return ham_->evecs() * res;
	}

	[[nodiscard]] bool can_merge(const Operator& rhs) const override
	{
		if(const auto* p = dynamic_cast<const HamEvol*>(&rhs))
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
