#pragma once
#include <Eigen/Sparse>

#include "Operators/Operator.hpp"

namespace qunn
{
class NonHermitianImpl 
{
private:
	Eigen::SparseMatrix<cx_double> op_;

public:
	explicit NonHermitianImpl(const Eigen::SparseMatrix<cx_double>& op)
		: op_{op}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit NonHermitianImpl(Eigen::SparseMatrix<cx_double>&& op)
		: op_{std::move(op)}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const
	{
		return op_*st;
	}

	void dagger_in_place()
	{
		op_ = op_.adjoint();
	}
};

class NonHermitian final
	: public Operator
{
private:
	std::shared_ptr<NonHermitianImpl> p_;

public:

	explicit NonHermitian(const Eigen::SparseMatrix<cx_double>& op)
		: Operator(op.rows()), p_{std::make_shared<NonHermitianImpl>(op)}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit NonHermitian(Eigen::SparseMatrix<cx_double>&& op)
		: Operator(op.rows()), 
		p_{std::make_shared<NonHermitianImpl>(std::move(op))}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit NonHermitian(const std::string& name,
			const Eigen::SparseMatrix<cx_double>& op)
		: Operator(op.rows(), name),
		p_{std::make_shared<NonHermitianImpl>(op)}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit NonHermitian(const std::string& name,
			Eigen::SparseMatrix<cx_double>&& op)
		: Operator(op.rows(), name),
		p_{std::make_shared<NonHermitianImpl>(std::move(op))}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	NonHermitian(const NonHermitian& rhs) = default;
	NonHermitian(NonHermitian&& rhs) = default;

	NonHermitian operator=(const NonHermitian& rhs) = delete;
	NonHermitian operator=(NonHermitian&& rhs) = delete;

	std::unique_ptr<Operator> clone() const override
	{
		auto copied = std::make_unique<NonHermitian>(*this);
		copied->set_name(name() + " copied");
		return copied;
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return p_->apply_right(st);
	}

	void dagger_in_place_impl() override
	{
		p_->dagger_in_place();
	}
};

}
