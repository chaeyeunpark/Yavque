#pragma once
#include <Eigen/Sparse>

#include "Operators/Operator.hpp"

namespace qunn
{
class SparseOperator final
	: public Operator
{
private:
	std::shared_ptr<Eigen::SparseMatrix<cx_double> > p_;

public:

	explicit SparseOperator(const Eigen::SparseMatrix<cx_double>& op)
		: Operator(op.rows()), 
		p_{std::make_shared<Eigen::SparseMatrix<cx_double>>(op)}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit SparseOperator(Eigen::SparseMatrix<cx_double>&& op)
		: Operator(op.rows()), 
		p_{std::make_shared<Eigen::SparseMatrix<cx_double>>(std::move(op))}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit SparseOperator(const Eigen::SparseMatrix<cx_double>& op,
			const std::string& name)
		: Operator(op.rows(), name),
		p_{std::make_shared<Eigen::SparseMatrix<cx_double>>(op)}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	explicit SparseOperator(Eigen::SparseMatrix<cx_double>&& op,
			const std::string& name)
		: Operator(op.rows(), name),
		p_{std::make_shared<Eigen::SparseMatrix<cx_double>>(std::move(op))}
	{
		assert(op.rows() == op.cols()); //check diagonal
	}

	SparseOperator(const SparseOperator& rhs) = default;
	SparseOperator(SparseOperator&& rhs) = default;

	SparseOperator operator=(const SparseOperator& rhs) = delete;
	SparseOperator operator=(SparseOperator&& rhs) = delete;

	std::unique_ptr<Operator> clone() const override
	{
		auto copied = std::make_unique<SparseOperator>(*this);
		copied->set_name(std::string("clone of ") + name());
		return copied;
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return (*p_)*st;
	}

	void dagger_in_place_impl() override
	{
		p_ = std::make_shared<Eigen::SparseMatrix<cx_double> >((*p_).adjoint());
	}
};
} //namespace qunn
