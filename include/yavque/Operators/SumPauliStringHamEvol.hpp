#pragma once

#include "../Univariate.hpp"
#include "../Variable.hpp"
#include "../utils.hpp"

#include "SumPauliString.hpp"

namespace yavque
{
class SumPauliStringHamEvol final : public Operator, public Univariate
{
private:
	bool conjugate_ = false;
	std::shared_ptr<const detail::SumPauliStringImpl> ham_;

	void dagger_in_place_impl() override { conjugate_ = !conjugate_; }

	SumPauliStringHamEvol(SumPauliStringHamEvol&&) = default;
	SumPauliStringHamEvol(const SumPauliStringHamEvol&) = default;

public:
	explicit SumPauliStringHamEvol(const SumPauliString& pstr)
		: Operator(pstr.dim(), "HamEvol of " + pstr.name()), ham_{pstr.get_impl()}
	{
		assert(pstr.mutually_commuting());
	}

	explicit SumPauliStringHamEvol(const SumPauliString& pstr, Variable var)
		: Operator(pstr.dim(), "HamEvol of " + pstr.name()),
		  Univariate(std::move(var)), ham_{pstr.get_impl()}
	{
		assert(pstr.mutually_commuting());
	}

	SumPauliStringHamEvol& operator=(const SumPauliStringHamEvol&) = delete;
	SumPauliStringHamEvol& operator=(SumPauliStringHamEvol&&) = delete;

	~SumPauliStringHamEvol() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto p = std::unique_ptr<SumPauliStringHamEvol>(new SumPauliStringHamEvol(*this));
		p->change_variable(Variable{var_.value()});
		return p;
	}

	[[nodiscard]] std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0., 1.0);
		std::string op_name = std::string("derivative of ") + name(); // change to fmt
		cx_double constant = conjugate_ ? I : -I;
		return std::make_unique<SumPauliString>(ham_, op_name, constant);
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0., 1.0);
		cx_double t = -I * var_.value();
		if(conjugate_)
		{
			t = -t;
		}
		return ham_->apply_exp(t, st);
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
