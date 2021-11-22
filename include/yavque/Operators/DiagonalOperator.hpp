#pragma once

#include "../utils.hpp"
#include "Operator.hpp"

namespace yavque
{

namespace detail
{
	class DiagonalOperatorImpl
	{
	private:
		Eigen::VectorXcd diag_op_;

	public:
		explicit DiagonalOperatorImpl(Eigen::VectorXcd diag_op)
			: diag_op_{std::move(diag_op)}
		{
		}

		explicit DiagonalOperatorImpl(Eigen::VectorXcd&& diag_op)
			: diag_op_{std::move(diag_op)}
		{
		}

		[[nodiscard]] const Eigen::VectorXcd& get_diag_op() const& { return diag_op_; }

		[[nodiscard]] Eigen::VectorXcd get_diag_op() && { return diag_op_; }

		[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const
		{
			return diag_op_.cwiseProduct(st);
		}

		void conjugate() { diag_op_ = diag_op_.conjugate().eval(); }
	};
} // namespace detail

class DiagonalOperator final : public Operator
{
private:
	std::shared_ptr<const detail::DiagonalOperatorImpl> p_;

	cx_double constant_ = 1.0;

	void dagger_in_place_impl() override
	{
		constant_ = std::conj(constant_);
		auto p = std::make_shared<detail::DiagonalOperatorImpl>(*p_);
		p->conjugate();
		p_ = std::move(p);
	}

public:
	explicit DiagonalOperator(const Eigen::VectorXcd& diag_op,
	                          const std::string& name = {})
		: Operator(diag_op.size(), name),
		  p_{std::make_shared<detail::DiagonalOperatorImpl>(diag_op)}
	{
	}

	explicit DiagonalOperator(std::shared_ptr<const detail::DiagonalOperatorImpl> p,
	                          const std::string& name = {}, cx_double constant = 1.0)
		: Operator(p->get_diag_op().size(), name), p_{std::move(p)}, constant_{constant}
	{
	}

	~DiagonalOperator() override = default;

	[[nodiscard]] std::shared_ptr<const detail::DiagonalOperatorImpl> get_impl() const
	{
		return p_;
	}

	DiagonalOperator(const DiagonalOperator&) = default;
	DiagonalOperator(DiagonalOperator&&) = default;

	DiagonalOperator& operator=(const DiagonalOperator&) = delete;
	DiagonalOperator& operator=(DiagonalOperator&&) = delete;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto cloned = std::make_unique<DiagonalOperator>(*this);
		return cloned;
	}

	[[nodiscard]] Eigen::VectorXcd get_diag_op() const { return p_->get_diag_op(); }

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return constant_ * p_->apply_right(st);
	}
};

} // namespace yavque
